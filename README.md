# 利用人體姿勢辨識技術建構運動復健輔助系統

本系統為一套復健輔助平台，結合攝影機串流、動作角度判斷、使用者資料記錄與分析，幫助復健者或年長者進行復健訓練，並可記錄每日動作表現與進度。
 
📷 即時攝影鏡頭串流

🏃‍♂️ 訓練模式（雙手平舉 / 抬腿 / 自訂動作 / 復健模式）

🔢 自動統計每日復健次數與角度最大值

📊 模式熱度圖、角度變化圖、次數折線圖

👤 用戶註冊、登入與資料紀錄

📁 圖片與影片處理分析


# 🎬 示意影片

 [![觀看影片示範](https://img.youtube.com/vi/xxzxte7LMyA/0.jpg)](https://www.youtube.com/watch?v=xxzxte7LMyA)
---

## 📌 功能介紹

### ✅ 使用者系統
- 使用者註冊 / 登入 / 登出
- 使用者個人資料管理

### ✅ 運動/復健訓練功能（復健模式）
- 透過攝影機擷取即時影像
- 使用 OpenCV / Mediapipe 擷取身體角度
- 設定復健動作目標角度
- 自動計算動作達成次數與最大或最小角度

### ✅ 活動紀錄與分析
- 記錄每次復健活動時間、模式與次數
- 支援每日/每月分析圖表
- 查詢不同動作的歷史角度變化

### ✅ 媒體處理
- 上傳圖像與影片
- 對上傳圖片與影片執行動作分析
- 只處理有效格式（如 mp4、jpg、png 等）

---

## 🧾 安裝與執行方式

### ✅ 安裝必要套件
參考https://github.com/yo-tong/Developing-Exercise-Programs-with-Human-Pose-Recognition-Using-MediaPipe/tree/main/%E7%92%B0%E5%A2%83%E5%AE%89%E8%A3%9D

---

# 📊 使用者活動紀錄系統資料庫

本專案為使用者活動追蹤系統的資料庫設計，主要用途為記錄使用者的登入資訊、活動歷程與個別動作設定，特別適合應用於復健系統或動作監測系統。

---
 

 


## 📁 資料表說明
請建立一個 MySQL 資料庫，並執行初始化 SQL 檔建立下列資料表：
### 1️⃣ `users` - 使用者資料表

儲存平台使用者帳號與登入憑證。

| 欄位名稱     | 資料型別        | 說明             |
|------------|----------------|------------------|
| id         | INT(11)        | 主鍵，自動遞增       |
| username   | VARCHAR(100)   | 使用者帳號         |
| password   | VARCHAR(255)   | 使用者密碼（需加密儲存） |

🔐 **索引與限制**：
- `id` 為主鍵
- `username` 建立索引，加速查詢

---

### 2️⃣ `user_activity` - 使用者活動紀錄表

用來紀錄使用者每次的活動紀錄，包含動作模式、次數與角度。

| 欄位名稱      | 資料型別       | 說明                  |
|-------------|----------------|-----------------------|
| id          | INT            | 主鍵，自動遞增           |
| user_id     | INT            | 使用者 ID，外鍵連接`users` 表，對應 `users.id`         |
| mode        | VARCHAR(255)   | 模式名稱（如：復健、雙手平舉等） |
| start_time  | DATETIME       | 活動開始時間             |
| end_time    | DATETIME       | 活動結束時間             |
| count       | INT            | 動作次數             |
| action_name | VARCHAR(255)   | （復健模式）活動名稱或部位          |
| body_angle  | FLOAT          | （復健模式）活動時的最大角度       |


🔗 **外鍵關聯**：
- `user_id` → `users(id)`（ON DELETE CASCADE）

---

### 3️⃣ `actions` - 動作設定表（復健模式） 

此表格記錄使用者設定的每個（復健模式）動作目標，例如期望的角度與方向，用於比對與分析。

| 欄位名稱         | 資料型別       | 說明                             |
|----------------|----------------|----------------------------------|
| id             | INT            | 主鍵，自動遞增                       |
| user_id        | INT            | 使用者 ID（對應 `users.id`）          |
| action_name    | VARCHAR(255)   | 復健動作名稱                           |
| angle_type     | VARCHAR(255)   | 復健角度類型，例如「右手肘夾角」、「右手肩膀」           |
| initial_angle  | INT            | 初始角度                           |
| goal_angle     | INT            | 目標角度                           |
| angle_direction| VARCHAR(10)    | `角度變大` 或 `角度變小` 表示角度目標       |
| created_at     | TIMESTAMP      | 建立時間                           |
| updated_at     | TIMESTAMP      | 資料更新時間（自動更新）                |

---

## 🧩 使用情境範例

1. 使用者註冊或登入後，系統會從 `users` 表讀取帳號資訊。
2. 每次執行活動時，系統會在 `user_activity` 中新增一筆紀錄。
3. 在（復健模式） 活動需要目標設定，則需先在 `actions` 表中建立動作設定，供後續比對與分析。

---

## 🛠 SQL 建立指令

```sql
-- 建立使用者資料表
CREATE TABLE users (
    id INT(11) NOT NULL AUTO_INCREMENT,
    username VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
    password VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
    PRIMARY KEY (id),
    INDEX (username)
);

-- 建立活動紀錄表
CREATE TABLE user_activity (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    mode VARCHAR(255),
    start_time DATETIME,
    end_time DATETIME,
    count INT,
    action_name VARCHAR(255) NULL,
    body_angle FLOAT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 建立動作設定表
CREATE TABLE actions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    action_name VARCHAR(255) NOT NULL,
    angle_type VARCHAR(255),
    initial_angle INT,
    goal_angle INT,
    angle_direction VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);




```

### ✅  執行主程式

```bash
python app.py
```
然後開啟瀏覽器進入：http://localhost:5000
## 🗂️ 專案結構
```
📁 your_project/
├── app.py                  # 主 Flask 應用
├── static/                 # 靜態資源（圖片、CSS、影片）
├── templates/              # HTML 模板
├── test_image/             # 圖片 
├── test_video/             # 影片 
├── 環境安裝/                # 環境設置
│   ├── environment.yml     # 套件清單
└── README.md               # 使用說明
```
# 🧠 使用技術

| 技術       | 說明                             |
|------------|----------------------------------|
| Flask      | Python Web 框架，用於後端邏輯與路由控制 |
| OpenCV     | 攝影機串流與影像處理             |
| Mediapipe  | 人體關節偵測與角度分析           |
| MySQL      | 資料儲存與查詢                   |
| JavaScript | 前端倒數計時、即時資料互動       |
| Bootstrap  | 響應式網頁設計 UI 框架           |
| Markdown   | 文件撰寫與格式化語法             |

> 🔎 **建議使用 Chrome 或 Firefox 瀏覽器** 以獲得最佳體驗與相容性（特別是即時串流與圖表互動功能）。
 
## 📷 復健動作偵測流程
- 使用者開啟攝影機

- 系統開始擷取角度，根據預設或設定的目標角度計分

- 60 秒自動結束或手動停止

- 將最大角度、動作次數記錄進資料庫

## ⚠️ 注意事項
- 使用前請確保攝影機設備可正常運作

- 建議使用 Chrome 或 Firefox 瀏覽器以獲得最佳效果

- 資料庫連線請確保使用者、密碼、權限設定正確

- 若影片無法播放，請確認影片格式或檔名是否正確

## 🙋‍♀️ 聯絡方式
如有報錯建議，請聯絡：

📧 tel147258@gmail.com
