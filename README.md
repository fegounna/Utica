# UTICA: Multi-Objective Self-Distillation Foundation Model Pretraining for Time Series Classification

📄 [Paper](https://arxiv.org/abs/2603.01348) &nbsp;|&nbsp; 🧠 [Weights](https://huggingface.co/fegounna/Utica/tree/main)

---

## 🚀 Usage

**Step 1 — Install**
```bash
pip install mantis-tsfm huggingface_hub
```

**Step 2 — Load**
```python
import torch
from huggingface_hub import hf_hub_download
from mantis.architecture import Mantis8M

device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = Mantis8M(device=device)
ckpt = hf_hub_download(repo_id="fegounna/Utica", filename="pytorch_model.bin")
backbone.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
backbone = backbone.to(device)
backbone.eval()
```

**Step 3 — Resize & Extract Features**
```python
import torch.nn.functional as F

x = torch.randn(8, 1, 1000)  # your time series: [batch, channels, time]
x = F.interpolate(x, size=512, mode='linear', align_corners=False) # resize to a multiple of 32 if necessary

with torch.no_grad():
    features = backbone(x)
```

---

## 📚 Citation
```bibtex
@misc{moakher2026utica,
  title={UTICA: Multi-Objective Self-Distillation Foundation Model Pretraining for Time Series Classification},
  author={Yessin Moakher and Youssef Attia El Hili and Vasilii Feofanov},
  year={2026},
  url={https://arxiv.org/abs/2603.01348},
}
```

