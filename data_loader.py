#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 data_loader.py - 数据加载与预处理模块
 神经重症患者迁移应激护理决策系统 V1.0
 
 功能说明:
   1. 从临床研究数据集加载患者和家属信息
   2. 数据清洗、缺失值处理和异常值检测
   3. 分类变量编码转换
   4. 特征标准化和归一化
   5. 数据质量评估报告生成
================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from config import (
    FEATURE_COLUMNS, CATEGORICAL_MAPPINGS, MODEL_DIR,
    FRSS_SCORE_RANGE, FCTI_SCORE_RANGE, SAS_THRESHOLD, SDS_THRESHOLD
)


class DataLoader:
    """
    数据加载与预处理类
    
    负责从原始数据集加载数据，进行数据清洗、特征编码和标准化，
    为后续的风险评估和决策引擎提供高质量的数据输入。
    """
    
    def __init__(self, model_dir=MODEL_DIR):
        """
        初始化数据加载器
        
        Parameters:
            model_dir: 模型和预处理器存储目录路径
        """
        self.model_dir = model_dir
        self.scaler = None
        self.encoders = {}
        self.feature_cols = FEATURE_COLUMNS
        self._load_preprocessors()
    
    def _load_preprocessors(self):
        """加载预训练的预处理器（标准化器和编码器）"""
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        encoders_path = os.path.join(self.model_dir, "encoders.pkl")
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = StandardScaler()
        
        if os.path.exists(encoders_path):
            with open(encoders_path, 'rb') as f:
                self.encoders = pickle.load(f)
    
    def load_dataset(self, filepath):
        """
        加载临床研究数据集
        
        Parameters:
            filepath: Excel数据文件路径
            
        Returns:
            DataFrame: 清洗后的数据集
        """
        df = pd.read_excel(filepath)
        df = self._clean_data(df)
        return df
    
    def _clean_data(self, df):
        """
        数据清洗处理
        
        处理流程:
        1. 去除重复记录
        2. 处理缺失值（数值型用中位数填充，分类型用众数填充）
        3. 异常值检测与标记
        4. 数据类型转换
        """
        # 去除重复
        df = df.drop_duplicates()
        
        # 数值型缺失值用中位数填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # 分类型缺失值用众数填充
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
        
        return df
    
    def encode_categorical(self, df):
        """
        分类变量编码转换
        
        将文本型分类变量转换为数值型编码，便于机器学习模型处理。
        编码映射关系:
        - Gender: 男=0, 女=1
        - Education: 初中及以下=0, 高中/中专=1, 大专=2, 本科及以上=3
        - Relationship: 配偶=0, 子女=1, 父母=2, 兄弟姐妹=3
        - Monthly_Income: <3000=0, 3000-5000=1, 5000-8000=2, >8000=3
        - Diagnosis: 脑出血=0, 脑梗死=1, 颅脑损伤=2, 蛛网膜下腔出血=3
        
        Parameters:
            df: 包含原始分类变量的DataFrame
            
        Returns:
            DataFrame: 编码后的DataFrame（新增编码列）
        """
        df_encoded = df.copy()
        
        for col, mapping in CATEGORICAL_MAPPINGS.items():
            if col in df_encoded.columns:
                encoded_col = f"{col}_encoded"
                df_encoded[encoded_col] = df_encoded[col].map(mapping)
                # 处理未映射的值
                df_encoded[encoded_col].fillna(0, inplace=True)
                df_encoded[encoded_col] = df_encoded[encoded_col].astype(int)
        
        return df_encoded
    
    def prepare_features(self, df):
        """
        特征准备：编码 + 标准化
        
        Parameters:
            df: 原始DataFrame
            
        Returns:
            tuple: (特征矩阵X_scaled, 原始DataFrame_encoded)
        """
        df_encoded = self.encode_categorical(df)
        
        # 确保所有特征列都存在
        available_cols = [c for c in self.feature_cols if c in df_encoded.columns]
        X = df_encoded[available_cols].values
        
        # 标准化
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return X_scaled, df_encoded
    
    def prepare_single_patient(self, patient_info):
        """
        准备单个患者数据用于预测
        
        Parameters:
            patient_info: dict, 包含患者基本信息的字典
                {
                    'Age': int,
                    'Gender': str ('男'/'女'),
                    'Education': str,
                    'Relationship': str,
                    'Monthly_Income': str,
                    'ICU_Stay_Days': int,
                    'Diagnosis': str,
                    'GCS_Score': int,
                    'FRSS_T0': float,
                    'FCTI_T0': float,
                    'SAS_T0': float,
                    'SDS_T0': float
                }
        
        Returns:
            numpy.ndarray: 标准化后的特征向量
        """
        # 构建DataFrame
        df_single = pd.DataFrame([patient_info])
        
        # 编码分类变量
        df_single = self.encode_categorical(df_single)
        
        # 确保所有特征列都存在
        X = np.zeros((1, len(self.feature_cols)))
        for i, col in enumerate(self.feature_cols):
            if col in df_single.columns:
                X[0, i] = df_single[col].values[0]
        
        # 标准化
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return X_scaled
    
    def validate_input(self, patient_info):
        """
        输入数据验证
        
        验证输入数据的有效性和完整性，检查:
        1. 必填字段是否存在
        2. 数值是否在合理范围内
        3. 分类变量取值是否合法
        
        Parameters:
            patient_info: dict, 患者信息字典
            
        Returns:
            tuple: (是否有效: bool, 错误信息列表: list)
        """
        errors = []
        
        # 必填字段检查
        required_fields = ['Age', 'Gender', 'Education', 'Relationship',
                          'Monthly_Income', 'ICU_Stay_Days', 'Diagnosis',
                          'GCS_Score', 'FRSS_T0', 'FCTI_T0', 'SAS_T0', 'SDS_T0']
        
        for field in required_fields:
            if field not in patient_info:
                errors.append(f"缺少必填字段: {field}")
        
        if errors:
            return False, errors
        
        # 数值范围验证
        age = patient_info.get('Age')
        if not (18 <= age <= 100):
            errors.append(f"年龄范围应为18-100岁，当前值: {age}")
        
        icu_days = patient_info.get('ICU_Stay_Days')
        if not (1 <= icu_days <= 365):
            errors.append(f"ICU住院天数范围应为1-365天，当前值: {icu_days}")
        
        gcs = patient_info.get('GCS_Score')
        if not (3 <= gcs <= 15):
            errors.append(f"GCS评分范围应为3-15分，当前值: {gcs}")
        
        frss = patient_info.get('FRSS_T0')
        if not (FRSS_SCORE_RANGE['min'] <= frss <= FRSS_SCORE_RANGE['max']):
            errors.append(f"FRSS评分范围应为{FRSS_SCORE_RANGE['min']}-{FRSS_SCORE_RANGE['max']}分，当前值: {frss}")
        
        fcti = patient_info.get('FCTI_T0')
        if not (FCTI_SCORE_RANGE['min'] <= fcti <= FCTI_SCORE_RANGE['max']):
            errors.append(f"FCTI评分范围应为{FCTI_SCORE_RANGE['min']}-{FCTI_SCORE_RANGE['max']}分，当前值: {fcti}")
        
        sas = patient_info.get('SAS_T0')
        if not (20 <= sas <= 100):
            errors.append(f"SAS评分范围应为20-100分，当前值: {sas}")
        
        sds = patient_info.get('SDS_T0')
        if not (20 <= sds <= 100):
            errors.append(f"SDS评分范围应为20-100分，当前值: {sds}")
        
        # 分类变量取值验证
        if patient_info.get('Gender') not in CATEGORICAL_MAPPINGS['Gender']:
            errors.append(f"性别取值错误，应为: {list(CATEGORICAL_MAPPINGS['Gender'].keys())}")
        
        if patient_info.get('Education') not in CATEGORICAL_MAPPINGS['Education']:
            errors.append(f"教育程度取值错误，应为: {list(CATEGORICAL_MAPPINGS['Education'].keys())}")
        
        if patient_info.get('Relationship') not in CATEGORICAL_MAPPINGS['Relationship']:
            errors.append(f"与患者关系取值错误，应为: {list(CATEGORICAL_MAPPINGS['Relationship'].keys())}")
        
        if patient_info.get('Monthly_Income') not in CATEGORICAL_MAPPINGS['Monthly_Income']:
            errors.append(f"月收入取值错误，应为: {list(CATEGORICAL_MAPPINGS['Monthly_Income'].keys())}")
        
        if patient_info.get('Diagnosis') not in CATEGORICAL_MAPPINGS['Diagnosis']:
            errors.append(f"诊断取值错误，应为: {list(CATEGORICAL_MAPPINGS['Diagnosis'].keys())}")
        
        return len(errors) == 0, errors
    
    def generate_quality_report(self, df):
        """
        生成数据质量评估报告
        
        Parameters:
            df: 数据集DataFrame
            
        Returns:
            dict: 数据质量报告
        """
        report = {
            "总记录数": len(df),
            "总字段数": len(df.columns),
            "缺失值统计": {},
            "数值型变量统计": {},
            "分类型变量分布": {}
        }
        
        # 缺失值统计
        missing = df.isnull().sum()
        report["缺失值统计"] = {col: int(count) for col, count in missing.items() if count > 0}
        
        # 数值型变量统计
        numeric_cols = ['Age', 'ICU_Stay_Days', 'GCS_Score', 'FRSS_T0', 'FCTI_T0', 'SAS_T0', 'SDS_T0']
        for col in numeric_cols:
            if col in df.columns:
                report["数值型变量统计"][col] = {
                    "均值": round(df[col].mean(), 2),
                    "标准差": round(df[col].std(), 2),
                    "最小值": df[col].min(),
                    "最大值": df[col].max(),
                    "中位数": df[col].median()
                }
        
        # 分类型变量分布
        categorical_cols = ['Gender', 'Education', 'Relationship', 'Diagnosis']
        for col in categorical_cols:
            if col in df.columns:
                report["分类型变量分布"][col] = df[col].value_counts().to_dict()
        
        return report
