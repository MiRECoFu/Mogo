from flask import Flask, request, jsonify, send_file
import os
from gen_t2m import gen_motions
from flask_cors import CORS
import random
import string
import oss2
from flask_socketio import SocketIO, emit
# import your_model_module  # 假设你的模型代码在这个模块中

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')
OSS_ACCESS_KEY_ID = ''
OSS_ACCESS_KEY_SECRET = ''
OSS_ENDPOINT = 'oss-cn-beijing.aliyuncs.com'  # 比如：oss-cn-hangzhou.aliyuncs.com
OSS_BUCKET_NAME = 'mogo-bvh'

auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, f"https://{OSS_ENDPOINT}", OSS_BUCKET_NAME)

# 设置 BVH 文件保存路径
SAVE_PATH = '/path/to/save/bvh/files/'

def generate_random_string(length=4):
    # 定义可能的字符集合，包括大写字母、小写字母和数字
    characters = string.ascii_letters + string.digits
    # 使用random.choice随机选择字符，并连接成指定长度的字符串
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

@app.route('/generate_motion', methods=['POST'])
def generate_motion():
    # 从请求中获取数据
    data = request.json
    prompt = data.get('prompt')
    length = data.get('length')
    dir_name = generate_random_string()
    
    if not prompt or not length:
        return jsonify({'error': 'Invalid input parameters'}), 400

    log_dir = './log'
    log_file = os.path.join(log_dir, 'prompts.txt')
    
    try:
        # 检查是否有 ./log 目录，没有则创建
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 以追加模式打开文件，写入 prompt
        with open(log_file, 'a') as f:
            f.write(f"{prompt}\n")
    
    except Exception as e:
        return jsonify({'error': f"Error logging prompt: {str(e)}"}), 500
    
    # 调用你的模型生成动作序列
    try:
        # bvh_path, save_path = gen_motions(prompt, length)
        bvh_path = gen_motions(prompt, length)
        oss_object_name = f'{dir_name}/{dir_name}.bvh'
        with open(bvh_path, 'rb') as file:
            bucket.put_object(oss_object_name, file, headers={'x-oss-object-acl': 'public-read'})
        oss_url = f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT}/{oss_object_name}"
        return jsonify({"oss_url": oss_url})
        # 检查文件是否存在
        # if not os.path.isfile(bvh_path):
        #     return jsonify({'error': 'Generated file not found'}), 500
        
        # # 返回 BVH 文件给前端
        # return send_file(
        #     bvh_path,
        #     as_attachment=True,
        #     download_name='output.bvh'
        # )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# WebSocket Handlers
@socketio.on('connect')
def handle_connect():
    emit('message', {'data': 'Connected to WebSocket server'})

@socketio.on('generate_motion_stream')
def handle_generate_motion(data):
    prompt = data.get('prompt')
    length = data.get('length')
    dir_name = generate_random_string()

    if not prompt or not length:
        emit('error', {'message': 'Invalid input parameters'})
        return

    try:
       gen_motions(prompt, length, emit)
        # socketio.start_background_task(target=gen_motions, prompt=prompt, length=length, socket=emit)

    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
