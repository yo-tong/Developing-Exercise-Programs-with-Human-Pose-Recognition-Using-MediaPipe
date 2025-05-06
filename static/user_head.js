// 點擊使用者頭像顯示/隱藏下拉選單
document
  .querySelector(".page-user-link") //選擇類名為 .page-user-link 的元素（通常是使用者頭像的連結）
  .addEventListener("click", function (event) {
    event.preventDefault(); // 阻止預設的點擊行為，例如連結跳轉
    const dropdownMenu = document.querySelector(".dropdown-menu");

    // 切換下拉選單的顯示狀態
    dropdownMenu.style.display =
      dropdownMenu.style.display === "block" ? "none" : "block";
  });

// 登出功能
function logout() {
  // 發送登出請求，跳轉到登出頁面
  window.location.href = "/logout"; // 將頁面重定向到登出路徑
}

// 點擊頁面其他地方時隱藏下拉選單
document.addEventListener("click", function (event) {
  const userMenu = document.querySelector(".user-menu"); // 獲取使用者選單元素
  const dropdownMenu = document.querySelector(".dropdown-menu"); // 獲取下拉選單元素

  // 如果點擊的地方不是使用者選單，則隱藏下拉選單
  if (!userMenu.contains(event.target)) {
    // 判斷點擊的區域是否在使用者選單內
    dropdownMenu.style.display = "none"; // 如果不是，則隱藏下拉選單
  }
});
function confirmLogout(event) {
  if (!confirm("確定要登出嗎？")) {
    // 彈出確認框，如果使用者選擇取消
    event.preventDefault(); // 阻止登出的表單提交
  }
}
function submitForm(event) {
  event.preventDefault(); // 阻止預設的提交行為
  document.querySelector("form").submit(); // 手動提交表單
}
