var http = require('http');
var fs=require('fs');
const urlLib=require('url');//处理url相关
var util = require('util');
http.createServer(function (request, response) {

    // response.writeHead(200, {'Content-Type': 'text/plain'});
    response.setHeader('Access-Control-Allow-Origin', '*'); //访问控制允许来源：所有
    response.setHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept'); //访问控制允许报头 X-Requested-With: xhr请求
    response.setHeader('Access-Control-Allow-Metheds', 'PUT, POST, GET, DELETE, OPTIONS'); //访问控制允许方法
    response.setHeader('X-Powered-By', 'nodejs'); //自定义头信息，表示服务端用nodejs
    // response.writeHead(200, {'Content-Type': 'text/plain'});
    var obj = urlLib.parse(request.url, true);
    var param = obj.query;
    console.log(param)
    var address = util.format('/mntc/yxy/MDPP/output/reprs/%s/figures/%s/%s_%s.json',
                                param["dataset"],param["mid"], param["mid"],param["order"])
    fs.readFile(address,
        'utf8',function (err, data) {
        if(err) console.log(err);
        response.end(data);
    });

}).listen(9618);


console.log('Server running at http://127.0.0.1:9618/');