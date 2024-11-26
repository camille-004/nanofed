# 🚀 NanoFed

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/camille-004/nanofed/ci.yml?style=for-the-badge)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nanofed?style=for-the-badge)
![Read the Docs](https://img.shields.io/readthedocs/nanofed?style=for-the-badge)
![GitHub License](https://img.shields.io/github/license/camille-004/nanofed?style=for-the-badge)
![PyPI - Status](https://img.shields.io/pypi/status/nanofed?style=for-the-badge)


**NanoFed**: *Simplifying the development of privacy-preserving distributed ML models.*

---

## 🌍 What is Federated Learning?

Federated Learning (FL) is a **distributed machine learning paradigm** that trains a global model across multiple clients (devices or organizations) without sharing their data. Instead, clients send model updates to a central server for aggregation.

### **Key Benefits**

| 🌟 Feature             | Description                                      |
|------------------------|--------------------------------------------------|
| 🔒 **Privacy Preservation** | Data stays securely on devices.                 |
| 🚀 **Resource Efficiency**   | Decentralized training reduces transfer overhead.|
| 🌐 **Scalable AI**           | Enables collaborative training environments.    |


---

## 📦 Installation

### **Requirements**

- Python `3.10+`
- Dependencies installed automatically

### **Install with Pip**

```bash
pip install nanofed
```

### **Development Installation**

```bash
git clone https://github.com/camille-004/nanofed.git
cd nanofed
make install
```

---

## 📖 Documentation

📚 **Learn how to use NanoFed in our guides and API references.**
👉 [**Read the Docs**](https://nanofed.readthedocs.io)

---

## ✨ Key Features

- 🔒 **Privacy-First**: Keep data on devices while training.
- 🚀 **Easy-to-Use**: Simple APIs with seamless PyTorch integration.
- 🔧 **Flexible**: Customizable aggregation strategies and extensible architecture.
- 💻 **Production Ready**: Robust error handling and logging.

### **Feature Overview**

| Feature                  | Description                                      |
|--------------------------|--------------------------------------------------|
| 🔒 **Privacy-First**      | Data never leaves devices.                       |
| 🚀 **Intuitive API**      | Built for developers with PyTorch support.       |
| 🔧 **Flexible Aggregation** | Supports custom strategies.                     |
| 💻 **Production Ready**   | Async communication, robust error handling.      |

---

## 🔧 Quick Start

Train a model using federated learning in just a few lines of code:

```python
import asyncio
from nanofed import HTTPClient, TorchTrainer, TrainingConfig

async def run_client(client_id: str, server_url: str):
    training_config = TrainingConfig(epochs=1, batch_size=256, learning_rate=0.1)
    async with HTTPClient(server_url, client_id) as client:
        model_state, _ = await client.fetch_global_model()
        await client.submit_update(model_state)

if __name__ == "__main__":
    asyncio.run(run_client("client1", "http://localhost:8080"))
