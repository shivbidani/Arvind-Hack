var express = require("express")

var app = express()

app.get("/list", (req, res, next) => {
	res.json({
		results:[{
		img:"http://dreamicus.com/data/face/face-04.jpg",
		bmi: "19"
	},{
		img: "http://oaksclan.com/wp-content/uploads/2017/07/unique-hairstyle-for-round-face-indian-male-hairstyle-for-fat-face-male-best-hairstyle-for-round-face-male.jpg",
		bmi: "21"
	 }
	]
	})
})

app.listen(process.env.PORT || 8080 )
