from flask import Flask, abort, request, jsonify
from package_API import one_user_recommend
app = Flask(__name__)

#测试数据暂时存放
tasks = []

@app.route('/add_task', methods=['POST'])
def add_task():
    if not request.json or 'User_id' not in request.json :
        abort(400)
    else:
        task = request.get_json().get('User_id')
        tasks.append(task)
    return jsonify({'result': 'User_id is posted successfully!'})

@app.route('/get_task', methods=['GET'])
def get_task():
    if not request.args  not in request.args:
        # 没有指定context则返回全部
        return jsonify(tasks)
    else:
        task_context = tasks[0]
        result = one_user_recommend().recommond_items(task_context)
        result_items = []
        for i in result.keys():
            result_items.append(i)
        return jsonify(result_items) if tasks else jsonify({'result': 'not found'})

if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(debug = True )