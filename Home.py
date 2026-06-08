#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Home.py - Streamlit Cloud部署入口文件
 神经重症患者迁移应激护理决策系统 V1.0
 
 部署说明:
   1. 将本文件夹推送到GitHub
   2. 在Streamlit Cloud (share.streamlit.io) 创建新应用
   3. 选择GitHub仓库和main分支
   4. Main file path: Home.py
   5. 点击Deploy
================================================================================
"""
import sys
import os

# 将src目录添加到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入主应用
from app import main

if __name__ == "__main__":
    main()
