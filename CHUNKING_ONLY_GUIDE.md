# Hướng dẫn sử dụng Vision-Based Chunking (Chỉ Chunking)

Tài liệu này hướng dẫn cách sử dụng phiên bản đơn giản của Vision-Based Chunking, chỉ thực hiện quá trình chunking mà không cần nhúng (embedding) và lưu trữ vào cơ sở dữ liệu.

## Các file mới được tạo

1. `src/simple_chunker.py`: Phiên bản đơn giản của `VisionChunker` chỉ thực hiện chunking
2. `simple_chunking.py`: Script để chạy quá trình chunking đơn giản

## Cách sử dụng

### Chuẩn bị môi trường

Đảm bảo bạn đã cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

### Chạy quá trình chunking đơn giản

```bash
python simple_chunking.py <đường_dẫn_đến_file_pdf> --output <thư_mục_đầu_ra>
```

Ví dụ:

```bash
python simple_chunking.py data/sample.pdf --output output
```

### Tham số tùy chọn

- `--batch-size`: Số trang mỗi batch (mặc định: 4)
- `--lmm-model`: Mô hình LMM sử dụng (mặc định: gemini-2.5-pro)
- `--output`: Thư mục đầu ra (mặc định: output)

### Ví dụ với các tùy chọn

```bash
python simple_chunking.py data/sample.pdf --output output --batch-size 2 --lmm-model gemini-2.5-pro
```

## Cấu trúc đầu ra

Kết quả chunking sẽ được lưu trong thư mục output dưới dạng các file JSON:

1. Các file `chunk_<id>.json` chứa thông tin chi tiết về từng chunk
2. File `chunks_summary.json` chứa tổng quan về tất cả các chunk

## Giải thích

Phiên bản đơn giản này sử dụng cùng các thuật toán chunking như phiên bản đầy đủ, nhưng không thực hiện quá trình nhúng và lưu trữ vào cơ sở dữ liệu, phù hợp cho các trường hợp chỉ cần phân đoạn tài liệu.
