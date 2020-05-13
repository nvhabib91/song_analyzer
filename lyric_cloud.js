let size = Math.min(Math.min(window.innerWidth, window.innerHeight), 600);
let color = d3.scaleOrdinal(d3.schemeCategory10);

let chart = d3.select("body")
	.append('svg')
		.attr("width", size)
		.attr("height", size);

let pack = d3.pack()
		.size([size, size])
		.padding(size*0.005);

d3.text('pokerface.txt', function(error, data) {
	if (error) throw error;

	let raw = data.toLowerCase()
	raw = raw.replace(/^'-' $/g, "").split(/[^'-\w]+/);

	let keys = [];

	let counts = raw.reduce(function(obj, word) {
		if(!obj[word]) {
			obj[word] = 0;
			keys.push(word);
		}
		obj[word]++;
		return obj;
	}, {});

	keys.sort(function(a,b) {
		return counts[b] - counts[a];
	});

	keys = keys.filter(function(key) {
	return counts[key] >= 5 ? key : '';
	});

	let root = d3.hierarchy({children: keys})
			.sum(function(d) { return counts[d]; });

	// console.log(root);

	let node = chart.selectAll(".node")
		.data(pack(root).leaves())
		.enter().append("g")
			.attr("class", "node")
			.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });

	node.append("circle")
			.attr("id", function(d) { return d.data; })
			.attr("r", function(d) { return d.r; })
			.style("fill", function(d) { return color(d.data); });

	node.append("clipPath")
			.attr("id", function(d) { return "clip-" + d.data; })
		.append("use")
			.attr("xlink:href", function(d) { return "#" + d.data; });

	node.append("text")
			.attr("clip-path", function(d) { return "url(#clip-" + d.data + ")"; })
		.append("tspan")
			.attr("x", 0)
			.attr("y", function(d) { return d.r/5; })
			.attr("font-size", function(d) { return d.r/3; })
			.text(function(d) { return d.data; });

});