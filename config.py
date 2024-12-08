# config.py

# 数据库配置
Hostname = 'localhost'
Port = 3306
Database = 'diss'
Username = 'root'
Password = 'lx110120119'

# 生成数据库 URI
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4'.format(Username, Password, Hostname, Port, Database)

# 配置 SQLAlchemy
SQLALCHEMY_DATABASE_URI = DB_URI
SQLALCHEMY_TRACK_MODIFICATIONS = False  # 禁用修改追踪功能，避免警告
