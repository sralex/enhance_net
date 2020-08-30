function post_multipart_json(path, data) {
    return post_multipart(path, data).then((response) => {
        return response.json()
    })
}

function post_multipart(path, data) {
    return window.fetch(path, {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
        },
        body: data
    });
}


var btn = document.getElementById('button');
var file = document.getElementById('file');
var name = document.getElementById('name');
var predict = document.getElementById('predict');
var img = document.getElementById('preview');
var img2 = document.getElementById('preview2');
var predict = document.getElementById('predict');
btn.addEventListener("click", function(evt){
	file.click()
	evt.preventDefault();
});
predict.addEventListener("click", function(evt){
  predict.style.display = "none";
	post_multipart_json("/predict",new FormData(document.getElementById("form"))).then(data => {
        img2.src = data["encoded"]
	}).catch(err => {
	    console.log({ err })
	})
	evt.preventDefault();
});
file.onchange = function(e) {
  predict.style.display = "inline-block";
  var file = e.target.files[0];
  img.src = URL.createObjectURL(event.target.files[0]);
  document.getElementById('name').innerHTML = "Description: </br>name: "+ file.name+" </br>size: "+file.size;
};