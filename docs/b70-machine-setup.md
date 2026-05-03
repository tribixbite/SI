# Intel B70 (Battlemage Pro) box — setup guide

Goal: stand up a dedicated **inference appliance** on the 2× Intel B70 cards so the daily-driver Qwen 3.6 LLM service can move off the SI workstation's 3090s. The 3090s then become exclusive to the SI training/anchor loop.

> **SKU note**: this guide uses "B70" as the user-supplied label. The most likely SKU is the Intel **Arc Pro B60** (24 GB Battlemage Pro, ~190 W). If your card is the rumored **Arc B770** (consumer Battlemage), substitute the model name in the driver/firmware checks but the rest of the recipe is identical. **Verify your card with `lspci | grep -i intel`** before installing — if anything below assumes B60 and your card disagrees, stop and confirm.

---

## 1. What this box does and doesn't do

**Does:**
- Host an OpenAI-compatible inference endpoint (port 8080 or similar) for the daily-driver LLM (Qwen 3.6 27B / 35B-A3B).
- Serve via `llama.cpp` (Vulkan or SYCL backend), `IPEX-LLM`, or `OpenVINO`.
- Replace the role currently filled by `~/llama-cpp-turboquant/llama-server` on the SI workstation's GPU 0.

**Doesn't:**
- Run SI training (Unsloth + bitsandbytes are CUDA-only). Training stays on the 3090.
- Run vLLM as we use it for SI anchoring (CUDA-only mainline).
- Boot Windows / WSL. Native Linux only.

---

## 2. OS choice

**Pick: Ubuntu 24.04 LTS with HWE (Hardware Enablement) kernel.**

Why:
- Intel's compute runtime (`intel-compute-runtime`, `intel-level-zero-gpu`) ships official `.deb` packages targeting Ubuntu LTS first. Other distros lag.
- Battlemage proper support landed in mainline kernel 6.10. Ubuntu 24.04 base ships 6.8, but the HWE stack (24.04.x with `linux-generic-hwe-24.04`) is currently 6.11+, which fully supports B60/B770. Don't run plain 24.04 base kernel.
- IPEX-LLM, OpenVINO, and `oneAPI` all have first-class Ubuntu 24.04 support.
- LTS = 5 years of security updates. Lower-friction than a rolling distro for an appliance.

Alternatives and why I'd skip them:
- **Ubuntu 26.04 LTS** (just released April 2026): drivers will be there but ecosystem maturity for AI tooling lags 6+ months after release. Wait until end of 2026.
- **Fedora 40/41**: bleeding-edge kernel is great for hardware support, but Intel's GPU tooling targets Ubuntu first; you'd be writing your own packaging.
- **Arch / rolling**: fine if you're hands-on, but every kernel bump can break Intel compute runtime's userspace until Intel publishes the matching package.
- **Debian 13**: stable but kernel will lag Battlemage support by 6–12 months.

---

## 3. Hardware checklist before installing

- [ ] **BIOS**: enable **Resizable BAR** (a.k.a. `Above 4G Decoding` on some boards). Without it, Battlemage performance is halved or worse.
- [ ] **PCIe**: each B60 wants Gen 4 ×8 minimum. Don't bifurcate down to ×4 unless you have to.
- [ ] **Power**: each B60 ~190 W TBP. 850 W PSU is comfortable for two; 1000 W if the CPU is also a power-hog.
- [ ] **Cooling**: B60s are blower-style and exhaust out the back — they fit dense systems but the rear of the case must vent freely.
- [ ] **CPU**: any modern x86_64 with AVX2; 8+ cores recommended. The cards do the work; the CPU just feeds them.
- [ ] **RAM**: 32 GB minimum. 64 GB if you intend to run two model instances concurrently.
- [ ] **Storage**: 500 GB NVMe minimum (200 GB for OS + tooling, ~50 GB per model variant in cache).

---

## 4. Install Ubuntu 24.04 LTS

1. Download **Ubuntu 24.04.x Server** ISO (not Desktop — this is a headless appliance).
   - https://releases.ubuntu.com/24.04/
2. Flash to USB with `dd` or `balenaEtcher`.
3. Boot the B70 box, install:
   - Hostname: e.g. `b70-infer`
   - Username: `matilda` (or whatever matches your SI box for shared SSH keys)
   - Software selection: **OpenSSH server**, no others.
   - Disk: full disk, LVM, no encryption (this is an appliance, not a laptop).
4. After first boot, SSH in from the SI box.

```bash
# from SI box
ssh matilda@b70-infer
```

5. **First commands on the new box:**

```bash
# update everything
sudo apt update && sudo apt full-upgrade -y

# install HWE kernel (gives us 6.11+, needed for Battlemage proper support)
sudo apt install -y --install-recommends linux-generic-hwe-24.04

# reboot into the HWE kernel
sudo reboot
```

After reboot:
```bash
uname -r            # should show 6.11.x or higher
lspci -nn | grep -i intel | grep -i vga    # confirm B70 detected
```

---

## 5. Intel GPU drivers + compute runtime

```bash
# Add Intel's APT repo for compute runtime (Level Zero, OpenCL)
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
https://repositories.intel.com/gpu/ubuntu noble unified" | \
    sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list

sudo apt update

# install compute runtime, level-zero, OpenCL ICD, and tools
sudo apt install -y \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    intel-media-va-driver-non-free \
    libmfx1 \
    libmfxgen1 \
    libvpl2 \
    libegl-mesa0 \
    libegl1-mesa \
    libgbm1 \
    libgl1-mesa-dri \
    libglapi-mesa \
    libgles2-mesa \
    libglx-mesa0 \
    libigdgmm12 \
    libxatracker2 \
    mesa-va-drivers \
    mesa-vdpau-drivers \
    mesa-vulkan-drivers \
    va-driver-all \
    vainfo \
    intel-gpu-tools

# add user to render + video groups (required to use the GPU as non-root)
sudo gpasswd -a $USER render
sudo gpasswd -a $USER video

# logout + login (or reboot) so group changes apply
sudo reboot
```

After reboot, verify:
```bash
# should show both B70s
clinfo -l

# real-time GPU usage tool (like nvtop for Intel)
sudo intel_gpu_top
```

---

## 6. Pick an inference stack

Three viable paths. Listed easiest → hardest, all tested-working as of 2026.

### Option A: `llama.cpp` with Vulkan backend (recommended — simplest, GGUF-friendly)

This is the most direct port of the SI box's existing `llama-cpp-turboquant` setup. Same model files (GGUF), same OpenAI-compatible HTTP API, just running on Intel via Vulkan instead of CUDA.

```bash
# install build deps
sudo apt install -y build-essential cmake git \
    libvulkan-dev vulkan-tools \
    libopenblas-dev pkg-config

# clone + build with Vulkan backend
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=1 -DGGML_OPENBLAS=1
cmake --build build --config Release -j$(nproc)

# quick sanity check
./build/bin/llama-cli --version

# pull a model from your SI box (over SSH)
mkdir -p ~/models
rsync -avP matilda@si-box:/home/matilda/models/Qwen3.6-27B-UD-Q4_K_XL.gguf ~/models/

# launch the server, pinned to both Vulkan devices
./build/bin/llama-server \
    -m ~/models/Qwen3.6-27B-UD-Q4_K_XL.gguf \
    --host 0.0.0.0 --port 8080 \
    --n-gpu-layers 999 \
    --ctx-size 32768 \
    -ngl 99 \
    --flash-attn \
    --jinja
```

To shard a 35B-A3B across both B70s, add `--tensor-split 50,50` (Vulkan backend supports multi-GPU as of late 2024).

For a `systemd` unit so it auto-starts:

```ini
# /etc/systemd/system/llama-server.service
[Unit]
Description=llama.cpp server (Qwen3.6-27B on B70)
After=network.target

[Service]
Type=simple
User=matilda
WorkingDirectory=/home/matilda/llama.cpp
ExecStart=/home/matilda/llama.cpp/build/bin/llama-server \
    -m /home/matilda/models/Qwen3.6-27B-UD-Q4_K_XL.gguf \
    --host 0.0.0.0 --port 8080 \
    -ngl 999 --ctx-size 32768 --flash-attn --jinja
Restart=on-failure
RestartSec=15

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
sudo journalctl -u llama-server -f
```

### Option B: IPEX-LLM (Intel's PyTorch fork, has its own vLLM-XPU)

Better feature parity with the CUDA stack (paged attention, continuous batching), worse hardware compatibility window (lags vanilla vLLM).

```bash
# install miniforge (lighter than full conda)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b
source ~/miniforge3/bin/activate

# create the IPEX-LLM env
conda create -n ipex-llm python=3.11 -y
conda activate ipex-llm

# install IPEX-LLM with XPU support
pip install --pre --upgrade ipex-llm[xpu] \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# install IPEX-LLM's vLLM fork
pip install --pre --upgrade ipex-llm-vllm[xpu] \
    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# verify XPU is visible
python -c "import torch; import intel_extension_for_pytorch; print(torch.xpu.is_available(), torch.xpu.device_count())"
# should print: True 2
```

Launch a vLLM-style server:
```bash
python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-27B-Instruct \
    --tensor-parallel-size 2 \
    --port 8080 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --quantization sym_int4
```

### Option C: OpenVINO + optimum-intel (Intel's official path)

Most efficient on Intel hardware, narrowest model support window. Use only if Options A/B underperform.

```bash
pip install openvino optimum[openvino,nncf]
optimum-cli export openvino --model Qwen/Qwen3.6-27B-Instruct --weight-format int4 ov_qwen3_6_27b
```

Then serve with OpenVINO Model Server (a separate Docker container).

---

## 7. Wiring it up to the SI box

Once the B70 box is serving on `http://b70-infer:8080`, redirect the SI workstation's daily-driver client (Open WebUI, LiteLLM, whatever's in `~/git/wsl-llm`) at the new endpoint:

```bash
# on SI box, in ~/git/wsl-llm
# edit ~/llama-server.conf or the LiteLLM config
# change LLAMA_BIN / endpoint URLs to point at b70-infer:8080
# stop the local llama-server systemd unit (it's now redundant)
sudo systemctl stop llama-server
sudo systemctl disable llama-server
```

Now both 3090s on the SI box are exclusively for SI work. Run the next anchor with `--tensor-parallel-size 2` (vLLM flag) to use both 3090s — this should resolve the cumulative `cudaErrorUnknown` we've been hitting on long Qwen3-Coder runs (two-card configs are typically more stable).

---

## 8. Smoke tests (verify before declaring victory)

```bash
# from SI box, hit the new B70 endpoint
curl http://b70-infer:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.6-27b",
        "messages": [{"role": "user", "content": "Write a Python one-liner that returns the sum of squares from 1 to n."}],
        "temperature": 0.2,
        "max_tokens": 128
    }' | jq -r '.choices[0].message.content'
```

Expected output: a one-line lambda or comprehension. If this comes back fast (<5s for 128 tokens), the B70 is producing real work.

```bash
# benchmark throughput (on B70 box)
sudo intel_gpu_top
# should show 60-90% engine utilization while a generation is running
```

---

## 9. Rollback if things go wrong

If the migration causes problems on the daily-driver workflow:
1. Re-enable the local `llama-server` systemd unit on the SI box (`sudo systemctl enable --now llama-server`).
2. Point Open WebUI / LiteLLM back at `localhost:8080`.
3. Leave the B70 box running for testing without it being on the critical path.

---

## 10. What to verify before committing the migration

- [ ] B70 box steady-state idle power draw < 100 W (otherwise something's misconfigured — Battlemage idles low).
- [ ] llama-server (or IPEX-LLM) serves Qwen 3.6 27B at ≥ 15 tok/s (single user). Better than that is bonus.
- [ ] Generation works for 1+ hour continuous without OOM or driver hangs.
- [ ] Both B70s show GPU activity under `intel_gpu_top` during multi-GPU load (otherwise tensor split isn't actually using both).
- [ ] SSH access from SI box works without password (key auth).
- [ ] Endpoint reachable from SI box's docker containers (the `litellm` proxy needs to hit it).

If all of those pass: flip the daily-driver endpoint, then `tensor-parallel-size 2` the SI vLLM. Should kill the `cudaErrorUnknown` runaway and double our anchor throughput simultaneously.
