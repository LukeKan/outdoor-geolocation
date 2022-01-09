(function (){
    const random_button = document.getElementById("random-button");



})()

$("#random-button").click(function(){
	let container = document.getElementById("demo-explained");
	let image = document.getElementById("pred-image");
	let gradcam = document.getElementById("gradcam-image");
	let top_pred = document.getElementById("top-confidence");
	console.log("Requesting random prediction...")
	$.ajax({
	  type: 'POST',
	  url: "/random",
	  error: function (request, status, error) {
			alert('Error: ' + error);
		 },
	  success: function(data){
	  	 data = JSON.parse(data);
		 image.src = data["img-name"];
	  	 gradcam.src = "/test_img/cam.jpg?"+ new Date().getTime();
		 $("#map").attr('src','/outdoor-geolocation/map/'+data['map-name']);
		 top_pred.textContent = data["top-confidence"];
		 container.style.visibility  = "visible";
	   }
	});
});