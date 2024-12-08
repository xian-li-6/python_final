import os
import json
import torch
from flask import Flask, render_template, redirect, url_for, request, flash, send_from_directory, session
from werkzeug.utils import secure_filename
from torchvision import models, transforms
from PIL import Image
from config import SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from exts import db
from SQLmodels import User
from flask_migrate import Migrate

# 创建 Flask 应用实例
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于 session 和 flash 消息

# 加载配置
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
app.config['UPLOAD_FOLDER'] = 'uploads'  # 上传的文件存放路径
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}  # 允许上传的文件类型
app.config['MODEL_PATH'] = './model/efficientnet_b7.pth'  # 模型路径

# 初始化数据库和迁移工具
db.init_app(app)
migrate = Migrate(app, db)

# 创建上传文件夹（如果不存在）
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 加载 EfficientNet-B7 模型
model = models.efficientnet_b7(pretrained=False)  # 不加载预训练权重
model.load_state_dict(torch.load(app.config['MODEL_PATH']))  # 加载模型权重
model.eval()  # 设置为评估模式

# 图像预处理操作
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载 ImageNet 类别标签（1000 类）
LABELS_FILE = 'imagenet_class_index.json'

# 如果没有下载标签文件，则下载
if not os.path.exists(LABELS_FILE):
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    import urllib.request
    urllib.request.urlretrieve(url, LABELS_FILE)

with open(LABELS_FILE) as f:
    class_idx = json.load(f)

# 从 class_idx 中提取标签
LABELS = [class_idx[str(i)][1] for i in range(1000)]

# 检查上传文件类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 根路径重定向到登录页面
@app.route('/')
def index():
    return redirect(url_for('login'))  # 确保首页是登录页

# 登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # 查找用户
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):  # 校验密码
            flash('登录成功', 'success')
            session['user_id'] = user.id  # 设置用户ID
            return redirect(url_for('home'))  # 登录成功后跳转到首页
        else:
            flash('用户名或密码错误', 'error')

    return render_template('login.html')

# 注册路由
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # 检查用户名和邮箱是否已经存在
        if User.query.filter_by(username=username).first():
            flash('用户名已存在', 'error')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('邮箱已存在', 'error')
            return redirect(url_for('register'))

        # 创建新用户
        new_user = User(username=username, email=email)
        new_user.set_password(password)  # 设置加密后的密码

        db.session.add(new_user)
        db.session.commit()

        flash('注册成功！请登录', 'success')
        return redirect(url_for('login'))  # 注册成功后跳转到登录页面

    return render_template('login.html')

# 退出路由
@app.route('/logout')
def logout():
    session.clear()  # 清除当前会话
    flash('退出成功', 'success')  # 提示退出成功
    return redirect(url_for('login'))  # 重定向到登录页面

# 首页路由
@app.route('/index')
def home():
    return render_template('index.html')

# 物体识别页面
@app.route('/recognition', methods=['GET', 'POST'])
def recognition():
    if request.method == 'POST':
        # 检查文件是否上传
        if 'file' not in request.files:
            flash('没有文件上传', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('没有选择文件', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 进行图像识别
            img = Image.open(filepath)
            img_tensor = preprocess(img).unsqueeze(0)  # 预处理并添加批次维度

            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted_idx = torch.max(outputs, 1)
                predicted_label = LABELS[predicted_idx.item()]

            # 返回识别结果
            return render_template('recognition_result.html', filename=filename, label=predicted_label)

    return render_template('recognition.html')

# 识别结果页面
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 用户管理页面
@app.route('/user_management', methods=['GET', 'POST'])
def user_management():
    if not session.get('user_id'):  # 如果用户未登录，重定向到登录页面
        flash('请先登录', 'error')
        return redirect(url_for('login'))

    page = request.args.get('page', 1, type=int)  # 获取当前页面，默认是第一页
    users_per_page = 25  # 每页显示25个用户

    # 获取分页后的用户列表，注意 error_out 参数作为关键字参数传递
    users_paginated = User.query.paginate(page=page, per_page=users_per_page, error_out=False)

    return render_template('user_management.html', users=users_paginated.items, pagination=users_paginated)

# 编辑用户信息
@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    user = User.query.get_or_404(user_id)

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']

        # 检查用户名或邮箱是否已被其他用户占用
        if User.query.filter_by(username=username).first() and user.username != username:
            flash('用户名已被占用', 'error')
            return redirect(request.url)

        if User.query.filter_by(email=email).first() and user.email != email:
            flash('邮箱已被占用', 'error')
            return redirect(request.url)

        user.username = username
        user.email = email

        db.session.commit()  # 提交更新到数据库
        flash('用户信息已更新', 'success')
        return redirect(url_for('user_management'))  # 重定向到用户管理页面

    return render_template('edit_user.html', user=user)

# 删除用户
@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)  # 删除用户
    db.session.commit()  # 提交删除操作
    flash('用户已删除', 'success')
    return redirect(url_for('user_management'))  # 重定向到用户管理页面

if __name__ == '__main__':
    app.run(debug=True)
