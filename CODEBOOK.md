# CODEBOOK - TỪ ĐIỂN DỮ LIỆU ĐIỆN THOẠI DI ĐỘNG

Tài liệu này giải thích chi tiết cấu trúc, kiểu dữ liệu, miền giá trị và ý nghĩa của các cột dữ liệu trong 3 file dữ liệu thuộc dự án: `phone_names.csv`, `feature_matrix.csv`, và `tag_matrix.csv`.

---

## MỤC LỤC
1. [Phần 1: Dữ liệu Gốc (phone_names.csv)](#1-phần-1-dữ-liệu-gốc-phone_namescsv)
2. [Phần 2: Ma trận Đặc trưng (feature_matrix.csv)](#2-phần-2-ma-trận-đặc-trưng-feature_matrixcsv)
3. [Phần 3: Ma trận Nhãn nhu cầu (tag_matrix.csv)](#3-phần-3-ma-trận-nhãn-nhu-cầu-tag_matrixcsv)

---

## 1. Phần 1: Dữ liệu từ `phone_names.csv`

Bảng dưới đây mô tả chi tiết các trường thông tin trong file `phone_names.csv`.

| Tên Cột (Column Name) | Kiểu Dữ Liệu (Data Type) | Miền Giá Trị (Value Range) | Ý Nghĩa & Giải Thích (Description) |
| :--- | :--- | :--- | :--- |
| **Name** | `str` | 445 giá trị duy nhất (ví dụ: 'apple iphone 5s', 'honor 200 pro') | Tên đầy đủ của mẫu điện thoại di động. |
| **Brand** | `str` | 12 thương hiệu khác nhau (ví dụ: 'apple', 'samsung', 'xiaomi', 'oppo') | Tên thương hiệu sản xuất thiết bị. |
| **Price** | `float64` | 1,990,000.0 - 64,990,000.0 | Giá bán thực tế của điện thoại bằng đơn vị Việt Nam Đồng (VND). |
| **Battery** | `float64` | 1,560.0 - 9,000.0 | Dung lượng pin của điện thoại, đơn vị tính là mAh. |
| **RAM_min** | `float64` | 1.0 - 16.0 | Dung lượng bộ nhớ RAM tối thiểu của các phiên bản (GB). |
| **ROM_min** | `float64` | 16.0 - 1,024.0 | Dung lượng bộ nhớ trong tối thiểu của các phiên bản (GB). |
| **RAM_max** | `float64` | 1.0 - 16.0 | Dung lượng bộ nhớ RAM tối đa có sẵn cho mẫu điện thoại đó (GB). |
| **ROM_max** | `float64` | 16.0 - 2,048.0 | Dung lượng bộ nhớ trong tối đa có sẵn cho mẫu điện thoại đó (GB). |

---

## 2. Phần 2: Dữ liệu từ `feature_matrix.csv`

Bảng dưới đây mô tả chi tiết các trường thông tin trong file `feature_matrix.csv`.

| Tên Cột (Column Name) | Kiểu Dữ Liệu (Data Type) | Miền Giá Trị (Value Range) | Ý Nghĩa & Giải Thích (Description) |
| :--- | :--- | :--- | :--- |
| **Price** | `float64` | 0.0 - 1.0 | Giá tiền đã được chuẩn hóa bằng phương pháp Min-Max. |
| **antutu_11** | `float64` | 0.0 - 1.0 | Điểm hiệu năng AnTuTu v11 đã được chuẩn hóa về khoảng [0, 1]. |
| **rear_mp_max** | `float64` | 0.0 - 1.0 | Độ phân giải camera sau lớn nhất đã được chuẩn hóa về khoảng [0, 1]. |
| **Screen Size** | `float64` | 0.0 - 1.0 | Kích thước màn hình đã được chuẩn hóa về khoảng [0, 1]. |
| **Battery** | `float64` | 0.0 - 1.0 | Dung lượng pin đã được chuẩn hóa về khoảng [0, 1]. |
| **RAM_min** | `float64` | 0.0 - 1.0 | Dung lượng RAM tối thiểu đã được chuẩn hóa về khoảng [0, 1]. |
| **ROM_min** | `float64` | 0.0 - 1.0 | Dung lượng bộ nhớ trong tối thiểu đã được chuẩn hóa về khoảng [0, 1]. |
| **front_mp** | `float64` | 0.0 - 1.0 | Độ phân giải camera trước đã được chuẩn hóa về khoảng [0, 1]. |
| **Refresh Rate** | `float64` | 0.0 - 1.0 | Tần số quét màn hình đã được chuẩn hóa về khoảng [0, 1]. |
| **rear_count** | `float64` | 0.0 - 1.0 | Số lượng camera sau đã được chuẩn hóa về khoảng [0, 1]. |
| **Brand_Raw_Other_Brand** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Thương hiệu khác ngoài danh sách chính. |
| **Brand_Raw_apple** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Thương hiệu Apple. |
| **Brand_Raw_honor** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Thương hiệu Honor. |
| **Brand_Raw_oppo** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Thương hiệu Oppo. |
| **Brand_Raw_realme** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Thương hiệu Realme. |
| **Brand_Raw_samsung** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Thương hiệu Samsung. |
| **Brand_Raw_xiaomi** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Thương hiệu Xiaomi. |
| **Hardware_Tier_Budget_Hardware** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Phần cứng thuộc phân khúc bình dân. |
| **Hardware_Tier_Flagship_Hardware** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Phần cứng thuộc phân khúc cao cấp (Flagship). |
| **Hardware_Tier_High_Mid_Hardware** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Phần cứng thuộc phân khúc cận cao cấp/tầm trung cao. |
| **OS_Name_Android** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Hệ điều hành Android. |
| **OS_Name_iOS** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Hệ điều hành iOS. |
| **Display_AMOLED** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Loại màn hình AMOLED. |
| **Display_IPS LCD** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Loại màn hình IPS LCD. |
| **Display_LCD** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Loại màn hình LCD. |
| **Display_OLED** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Loại màn hình OLED. |
| **Display_Other** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Loại màn hình khác. |
| **NFC_0** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Không hỗ trợ công nghệ NFC. |
| **NFC_1** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Có hỗ trợ công nghệ NFC. |
| **rear_ois_0.0** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Không có chống rung quang học OIS cho camera sau. |
| **rear_ois_1.0** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Có chống rung quang học OIS cho camera sau. |
| **rear_wide_0.0** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Không có camera góc rộng phía sau. |
| **rear_wide_1.0** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Có camera góc rộng phía sau. |
| **rear_telephoto_0.0** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Không có camera telephoto (zoom quang học) phía sau. |
| **rear_telephoto_1.0** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Có camera telephoto (zoom quang học) phía sau. |
| **has_eSIM_0** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Không hỗ trợ eSIM. |
| **has_eSIM_1** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Có hỗ trợ eSIM. |
| **SIM_total_0** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Thiết bị không sử dụng SIM vật lý. |
| **SIM_total_1** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Thiết bị hỗ trợ tối đa 1 khe SIM. |
| **SIM_total_2** | `float64` | 0.0 hoặc 1.0 | One-hot encoding: Thiết bị hỗ trợ tối đa 2 khe SIM. |

---

## 3. Phần 3: Dữ liệu từ `tag_matrix.csv`

Bảng dưới đây mô tả chi tiết các trường thông tin trong file `tag_matrix.csv`.

| Tên Cột (Column Name) | Kiểu Dữ Liệu (Data Type) | Miền Giá Trị (Value Range) | Ý Nghĩa & Giải Thích (Description) |
| :--- | :--- | :--- | :--- |
| **Gaming_Need** | `int64` | 0 hoặc 1 | Gắn thẻ nhu cầu chơi game (1: Cấu hình phù hợp gaming, 0: Không). |
| **Battery_Need** | `int64` | 0 hoặc 1 | Gắn thẻ nhu cầu pin trâu (1: Thời lượng sử dụng pin dài, 0: Không). |
| **Photography_Need** | `int64` | 0 hoặc 1 | Gắn thẻ nhu cầu chụp ảnh (1: Hệ thống camera chất lượng cao, 0: Không). |
| **Performance_Need** | `int64` | 0 hoặc 1 | Gắn thẻ nhu cầu hiệu năng cao (1: Xử lý tác vụ nặng mượt mà, 0: Không). |
| **Large_Display_Need** | `int64` | 0 hoặc 1 | Gắn thẻ nhu cầu màn hình lớn (1: Không gian hiển thị rộng rãi, 0: Không). |
| **HighRes_Need** | `int64` | 0 hoặc 1 | Gắn thẻ nhu cầu độ phân giải cao (1: Màn hình sắc nét cao, 0: Không). |
| **Multitask_Need** | `int64` | 0 hoặc 1 | Gắn thẻ nhu cầu đa nhiệm (1: RAM lớn hỗ trợ chạy nhiều app, 0: Không). |
| **Budget_King_Need** | `int64` | 0 hoặc 1 | Gắn thẻ phân khúc vua giá rẻ (1: Giá cực tốt so với công năng, 0: Không). |
| **Midrange_Value_Need** | `int64` | 0 hoặc 1 | Gắn thẻ phân khúc tầm trung đáng mua (1: Cân bằng tốt giữa giá và hiệu năng, 0: Không). |
| **Premium_Luxury_Need** | `int64` | 0 hoặc 1 | Gắn thẻ phân khúc cao cấp/sang trọng (1: Flagship hoặc dòng siêu sang, 0: Không). |

---
