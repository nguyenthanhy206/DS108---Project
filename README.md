# Hệ Thống Khuyến Nghị Smartphone Đa Nguồn & Dashboard Phân Tích

Hệ thống khuyến nghị Smartphone toàn diện sử dụng kiến trúc dữ liệu 3 tầng, tích hợp dữ liệu đa nguồn từ các nền tảng thương mại và phần cứng danh tiếng (CellphoneS, GSMarena, Nanoreview). Hệ thống thực hiện bộ lọc lai (Lọc cứng theo nhu cầu + Lọc mềm bằng toán học hình học góc Cosine) đi kèm giao diện Dashboard tương tác trực quan hóa bằng Streamlit.

---

## 1. Kiến Trúc Dữ Liệu 3 Tầng (Data Architecture)

Hệ thống quản lý dữ liệu theo cấu trúc 3 tầng tách biệt nhằm tối ưu hóa hiệu năng tính toán và khả năng phản hồi thời gian thực của giao diện:
* **Tầng hiển thị (`df_cleaned.csv`):** Lưu trữ thông tin tường minh dạng chuỗi văn bản và số thực tế phục vụ hiển thị lên giao diện (Tên máy, hãng, giá bán, thông số cứng thô).
* **Tầng lọc thô (`tag_matrix.csv`):** Ma trận nhị phân (0/1) của các nhãn nhu cầu người dùng, đóng vai trò là bộ lọc cứng (Hard-Filter) loại bỏ ngay lập tức các máy không khớp tiêu chí.
* **Tầng tính toán (`feature_matrix.csv`):** Ma trận số thực kích thước $445 \times 40$ đã chuẩn hóa MinMax và xử lý giới hạn biên bằng `.clip(0, 1)`, đóng vai trò làm không gian vector toán học để tính độ tương đồng Cosine.

---

## 2. Yêu Cầu Hệ Thống & Cài Đặt (Prerequisites & Installation)

### a. Yêu cầu môi trường
* Python 3.10+
* Trình duyệt Google Chrome (Bắt buộc để chạy Selenium / Undetected-Chromedriver)

### b. Cài đặt thư viện
```bash
git clone <link-repo-cua-ban>
cd <DS108---Project>
pip install -r requirements.txt

```

*Các thư viện lõi bao gồm: streamlit, plotly, pandas, numpy, scikit-learn, requests, beautifulsoup4, undetected-chromedriver, scrapingbee.*

---

## 3. Hướng Dẫn Vận Hành Từ A Đến Z (Pipeline Execution Guide)

Để hệ thống vận hành chính xác và đồng bộ, bạn cần thực hiện nghiêm ngặt theo quy trình 3 bước dưới đây:

### 🔹 Bước 1: Thu thập dữ liệu thô (Data Scraping & Alignment)

Khởi chạy 3 script độc lập nằm trong thư mục `notebooks/` để thu thập dữ liệu từ 3 nguồn:

1. **Cào dữ liệu danh mục thị trường Việt Nam:**
```bash
python notebooks/cellphones_scraper.py

```


*Đầu ra:* Sinh ra file danh mục gốc `cellphones_raw.csv`.
2. **Cào dữ liệu hiệu năng toàn cầu từ Nanoreview:**
```bash
python notebooks/antutu_score_scraper.py

```


*Đầu ra:* Sinh ra file hiệu năng `antutu_socket.csv`.
3. **Cào dữ liệu cấu hình chi tiết phần cứng từ GSMarena:**
```bash
python notebooks/gsmarena_scraping.py

```


*Đầu ra:* Sinh ra các file dữ liệu cấu hình thô theo từng thương hiệu di động độc lập.

>  **HƯỚNG DẪN ĐỒNG BỘ DỮ LIỆU GSMARENA (Giải pháp gộp dữ liệu):**
> Do dữ liệu GSMarena được cào riêng lẻ theo từng hãng và chứa cả các model quốc tế không kinh doanh tại Việt Nam, bạn cần thực hiện gộp dữ liệu để định hình lại file `GSMarena.csv` chuẩn:
> * **Cách thực hiện:** Tiến hành hợp nhất (Merge) các file cấu hình thô của GSMarena (file `specs`) với cột `name` của file danh mục `cellphones_raw.csv` thông qua cơ chế phương thức giao thoa `how='inner'`.
> * **Mục đích:** Việc này đóng vai trò như một bộ lọc, giúp **chỉ giữ lại những dòng sản phẩm vừa có tên trong danh mục kinh doanh thực tế, vừa có đầy đủ thông số phần cứng chi tiết bên GSMarena**, loại bỏ hoàn toàn các máy rác không thu thập được thông tin để tạo ra file `GSMarena.csv` tinh gọn cuối cùng.
> 
> 

###  Bước 2: Tiền xử lý dữ liệu và Tạo ma trận 3 tầng (Data Preprocessing)

Sau khi đã chuẩn bị đủ 3 file thô sạch ở Bước 1 (`cellphones_raw.csv`, `antutu_socket.csv`, `GSMarena.csv`), hãy mở và chạy tuần tự các tài nguyên trong thư mục `notebooks/`:

1. **Chạy `preprocess_first.ipynb`:** Nhập module xử lý chuỗi từ `cleaning.py` để làm sạch chuỗi văn bản, trích xuất đơn vị phần cứng, xử lý trùng lặp. Đồng thời thực hiện **bù đắp chéo dữ liệu khuyết thiếu (Missing Data Imputation)** bằng cách lấy thông số từ `GSMarena.csv` và `antutu_socket.csv` điền vào các ô trống của `cellphones_raw.csv`.
2. **Chạy `preprocess_second.ipynb`:** Thực hiện kỹ nghệ đặc trưng (Feature Engineering), dán nhãn nhu cầu nhị phân, chuẩn hóa dữ liệu số thực về dải $[0, 1]$ và áp dụng `.clip(0, 1)` để tránh sai số dấu phẩy động.
* *Kết quả:* Sinh ra bộ ba file tầng dữ liệu đích đặt vào thư mục `data/processed/` (`df_cleaned.csv`, `tag_matrix.csv`, `feature_matrix.csv`).



###  Bước 3: Khởi chạy Ứng dụng Dashboard Streamlit

Vận hành giao diện hệ khuyến nghị và trình diễn phân tích:

```bash
streamlit run app.py

```

---

##  4. Các Tính Năng Chính Trên Dashboard

* **Tab Hệ Khuyến Nghị:**
* *Hard-Filter:* Bộ lọc cứng nhị phân dựa trên `tag_matrix.csv` giúp người dùng chọn nhanh nhu cầu cốt lõi (Gaming, Pin trâu, Nhiếp ảnh, Phân khúc ngân sách...).
* *Soft-Filter:* Tinh chỉnh thanh trượt trọng số (Slider) cá nhân hóa để chạy thuật toán so sánh hình học góc Cosine, tìm ra Top 5 sản phẩm tương đồng nhất.
* *Trực quan hóa:* So sánh tương quan cấu hình phần cứng của Top 5 máy thông qua biểu đồ Radar Chart tương tác của Plotly.


* **Tab Nghiên Cứu Khí Tượng / Phân Tích EDA:** Tích hợp báo cáo phân tích nâng cao, hiển thị phân phối mật độ dữ liệu, xử lý mất cân bằng nhãn và trực quan hóa các đặc trưng xu hướng (Trend/Delta) phục vụ bài toán mở rộng.

---

##  5. Cấu Trúc Thư Mục Dự Án (Project Structure)

Toàn bộ tài nguyên mã nguồn và dữ liệu của dự án được phân rã tường minh theo cấu trúc module hóa dưới đây:

```text
├── data/
│   ├── raw/                  # Chứa 3 file dữ liệu thô ban đầu sau khi cào
│   │   ├── cellphones_raw.csv
│   │   ├── antutu_socket.csv
│   │   └── GSMarena.csv      # File kết quả sau khi merge lọc theo tên máy CellphoneS
|   |   └── specs/            # Chứa tất cả các file thông số của các điện thoại lấy từ GSMarena (chia theo từng hãng)
│   └── processed/            # Bộ ba file cấu trúc 3 tầng hoàn chỉnh sau tiền xử lý
│       ├── df_cleaned.csv    # Tầng hiển thị giao diện
│       ├── tag_matrix.csv    # Tầng lọc thô (Hard-Filter)
│       └── feature_matrix.csv# Tầng tính toán toán học (Soft-Filter)
│
├── notebooks/                # Nơi lưu trữ toàn bộ mã nguồn xử lý và phân tích dữ liệu
│   ├── cellphones_scraper.py # Script cào dữ liệu danh mục từ CellphoneS
│   ├── gsmarena_scraping.py  # Script cào dữ liệu cấu hình từ GSMarena
│   ├── antutu_score_scraper.py# Script cào dữ liệu hiệu năng từ Nanoreview
│   ├── cleaning.py           # Module chứa các hàm làm sạch chuỗi và trích xuất số thô
│   ├── preprocess_first.ipynb# Notebook tiền xử lý tầng 1 & bù đắp dữ liệu khuyết thiếu
│   ├── preprocess_second.ipynb# Notebook kỹ nghệ đặc trưng & phân rã ma trận 3 tầng
│   ├── EDA_first.ipynb       # Notebook phân tích khám phá dữ liệu đợt 1
│   └── EDA_second.ipynb      # Notebook phân tích khám phá dữ liệu chuyên sâu đợt 2
│
├── app.py                    # File khởi chạy ứng dụng Dashboard Streamlit chính
├── CODEBOOK.md               # Tài liệu định nghĩa từ điển dữ liệu và ý nghĩa các biến
├── requirements.txt          # Danh sách các thư viện Python cần cài đặt
└── README.md                 # Tài liệu hướng dẫn vận hành dự án này

```

---

##  6. Một Số Lưu Ý Quan Trọng (Troubleshooting & Notes)

* **Lỗi API Key:** Làm mới/ Thay đổi API Key của ScrapingBee trong file `gsmarena_scraping.py` trước khi chạy tiến trình cào dữ liệu.
* **Lỗi Driver Selenium:** Nếu `undetected-chromedriver` trong script Nanoreview báo lỗi, hãy kiểm tra lại phiên bản Google Chrome hiện tại trên thiết bị để đảm bảo tính tương thích và cập nhật `version_main` trong code.
* **Xử lý NaN toán học:** Quy trình xử lý missing data đã được tự động hóa. Nếu gặp lỗi toán học dạng bộ nhớ hoặc phép chia cho không ($0/0$) phát sinh khi chuẩn hóa ma trận đặc trưng, hệ thống sẽ tự động vá bằng `.fillna(0)`.

```

```
