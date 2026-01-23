// 1. Use querySelector to get a SINGLE element (getElementsByClassName returns a list)
let clickbutton = document.querySelector(".Smart-Search");

// 2. Check if the button exists on this specific page before using it
if (clickbutton) {
    clickbutton.addEventListener("click", function () {
        // 3. Use window.location to redirect in the browser. 
        // 'res.render' is a SERVER command and cannot be used here.
        window.location.href = "/ai"; 
        console.log("click");
    });
}