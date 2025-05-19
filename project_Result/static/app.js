let header = document.querySelector("header");

window.addEventListener("scroll", () => {
  if (window.scrollY == 0) {
    header.style.boxShadow = "";
  } else {
    header.style.boxShadow = "0 10px 6px -6px #777";
  }
});
