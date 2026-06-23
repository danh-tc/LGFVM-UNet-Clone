Những gì đã được tạo
requirements.txt — nguồn sự thật
Chỉ giữ đúng những package code thực sự import (~30 package thay vì 140+ package gốc đầy rác)
Có --extra-index-url PyTorch ở đầu file (hỗ trợ natively bởi pip)
Pinned các version quan trọng:
timm==0.4.12 — bắt buộc, code dùng timm.models.layers (API cũ, bị remove ở timm mới hơn)
transformers==4.46.3 — bắt buộc, mamba-ssm 2.1.0 import API đã bị xóa ở transformers 5.x
torch==2.3.1+cu121, causal-conv1d==1.4.0, mamba-ssm==2.1.0
install.sh — 1 lệnh duy nhất

bash install.sh
Script tự động hóa 3 bước phải làm theo đúng thứ tự:

Cài torch với --index-url riêng của PyTorch
Compile mamba-ssm với --no-build-isolation (để thấy torch đã cài)
pip install -r requirements.txt cho phần còn lại
Tại sao không thể chỉ pip install -r requirements.txt một bước?
Vấn đề	Lý do
torch==2.3.1+cu121	Không có trên PyPI, chỉ có trên download.pytorch.org/whl/cu121
mamba-ssm, causal-conv1d	Cần compile CUDA — pip tạo isolated build env không thấy torch, phải dùng --no-build-isolation
