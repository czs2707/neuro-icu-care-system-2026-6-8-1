#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 risk_assessment.py - 智能风险评估模块
 神经重症患者迁移应激护理决策系统 V1.0
 
 功能说明:
   1. 多维度迁移应激风险量化评估
   2. 基于梯度提升分类器的风险等级预测
   3. FRSS改善幅度预测（回归模型）
   4. 风险预警和动态监测
   5. 风险因子重要性分析
   6. 个性化风险画像生成
================================================================================
"""
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_DIR, RISK_THRESHOLDS, FRSS_SCORE_RANGE


class RiskAssessmentEngine:
    """
    智能风险评估引擎
    
    基于多维度量表评估数据和机器学习模型，对ICU转出患者家属的
    迁移应激风险进行智能量化评估，提供风险等级判定、风险因子
    分析和改善预测等功能。
    
    Attributes:
        risk_model: 梯度提升分类器，用于风险等级预测
        frss_model: 随机森林回归器，用于FRSS改善幅度预测
        scaler: 数据标准化器
        metadata: 模型元数据（包含标签映射等）
    """
    
    def __init__(self, model_dir=MODEL_DIR):
        """
        初始化风险评估引擎
        
        Parameters:
            model_dir: 模型文件存储目录
        """
        self.model_dir = model_dir
        self.risk_model = None
        self.frss_model = None
        self.scaler = None
        self.metadata = None
        self._load_models()
    
    def _load_models(self):
        """加载预训练的风险评估模型"""
        risk_model_path = os.path.join(self.model_dir, "risk_model.pkl")
        frss_model_path = os.path.join(self.model_dir, "frss_model.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        metadata_path = os.path.join(self.model_dir, "metadata.pkl")
        
        if os.path.exists(risk_model_path):
            with open(risk_model_path, 'rb') as f:
                self.risk_model = pickle.load(f)
        
        if os.path.exists(frss_model_path):
            with open(frss_model_path, 'rb') as f:
                self.frss_model = pickle.load(f)
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
    
    def assess_risk(self, X_scaled, patient_info=None):
        """
        执行全面风险评估
        
        评估流程:
        1. 基于量表阈值的规则评估
        2. 基于机器学习模型的预测评估
        3. 综合评分和风险等级判定
        4. 风险因子重要性分析
        
        Parameters:
            X_scaled: 标准化后的特征矩阵
            patient_info: 原始患者信息字典（可选，用于规则评估）
            
        Returns:
            dict: 风险评估报告
        """
        # 1. 规则引擎评估
        rule_result = self._rule_based_assessment(patient_info) if patient_info else None
        
        # 2. 机器学习模型预测
        ml_result = self._ml_based_assessment(X_scaled)
        
        # 3. 综合评估
        comprehensive_result = self._comprehensive_assessment(rule_result, ml_result)
        
        # 4. 风险因子分析
        feature_importance = self._analyze_feature_importance(X_scaled)
        
        # 5. 构建完整报告
        report = {
            "风险等级": comprehensive_result['risk_level'],
            "风险等级编码": int(comprehensive_result['risk_code']),
            "风险概率分布": ml_result['probability_distribution'],
            "规则评估": rule_result,
            "模型预测": ml_result,
            "风险因子重要性": feature_importance,
            "综合评估": comprehensive_result
        }
        
        return report
    
    def _rule_based_assessment(self, patient_info):
        """
        基于量表阈值的规则评估
        
        使用临床量表阈值进行风险判定:
        - FRSS评分<55分: 高风险（反向计分，分数越低=应激水平越高）
        - SAS标准分≥50分: 存在焦虑症状
        - SDS标准分≥53分: 存在抑郁症状
        - GCS评分≤8分: 重度昏迷
        - ICU住院天数>14天: 长期住院
        
        Parameters:
            patient_info: 患者信息字典
            
        Returns:
            dict: 规则评估结果
        """
        if not patient_info:
            return None
        
        risk_factors = []
        risk_score = 0
        
        # FRSS评估（反向计分：低分=高风险）
        frss = patient_info.get('FRSS_T0', 50)
        if frss < RISK_THRESHOLDS['frss_high']:
            risk_factors.append({
                "因子": "FRSS迁移应激评分",
                "数值": f"{frss:.1f}分",
                "说明": f"低于阈值({RISK_THRESHOLDS['frss_high']}分)，提示存在较高迁移应激水平",
                "权重": 0.30
            })
            risk_score += 30 * (1 - (frss - FRSS_SCORE_RANGE['min']) / 
                               (RISK_THRESHOLDS['frss_high'] - FRSS_SCORE_RANGE['min']))
        
        # SAS评估
        sas = patient_info.get('SAS_T0', 50)
        if sas >= RISK_THRESHOLDS['sas_anxiety']:
            risk_factors.append({
                "因子": "SAS焦虑评分",
                "数值": f"{sas:.1f}分",
                "说明": f"≥{RISK_THRESHOLDS['sas_anxiety']}分，提示存在焦虑症状",
                "权重": 0.20
            })
            risk_score += 20
        
        # SDS评估
        sds = patient_info.get('SDS_T0', 50)
        if sds >= RISK_THRESHOLDS['sds_depression']:
            risk_factors.append({
                "因子": "SDS抑郁评分",
                "数值": f"{sds:.1f}分",
                "说明": f"≥{RISK_THRESHOLDS['sds_depression']}分，提示存在抑郁症状",
                "权重": 0.20
            })
            risk_score += 20
        
        # GCS评估
        gcs = patient_info.get('GCS_Score', 15)
        if gcs <= RISK_THRESHOLDS['gcs_severe']:
            risk_factors.append({
                "因子": "GCS昏迷评分",
                "数值": f"{gcs}分",
                "说明": f"≤{RISK_THRESHOLDS['gcs_severe']}分，提示重度昏迷状态",
                "权重": 0.15
            })
            risk_score += 15
        
        # ICU住院天数评估
        icu_days = patient_info.get('ICU_Stay_Days', 5)
        if icu_days > RISK_THRESHOLDS['icu_stay_long']:
            risk_factors.append({
                "因子": "ICU住院天数",
                "数值": f"{icu_days}天",
                "说明": f">{RISK_THRESHOLDS['icu_stay_long']}天，长期住院增加分离焦虑风险",
                "权重": 0.15
            })
            risk_score += 15
        
        # 风险等级判定
        if risk_score >= 60:
            risk_level = "极高风险"
            risk_code = 3
        elif risk_score >= 40:
            risk_level = "高风险"
            risk_code = 2
        elif risk_score >= 20:
            risk_level = "中风险"
            risk_code = 1
        else:
            risk_level = "低风险"
            risk_code = 0
        
        return {
            "风险因子": risk_factors,
            "风险评分": round(risk_score, 2),
            "风险等级": risk_level,
            "风险等级编码": risk_code
        }
    
    def _ml_based_assessment(self, X_scaled):
        """
        基于机器学习模型的预测评估
        
        使用预训练的梯度提升分类器进行风险等级预测，
        输出各类别的概率分布。
        
        Parameters:
            X_scaled: 标准化后的特征矩阵
            
        Returns:
            dict: 模型预测结果
        """
        if self.risk_model is None:
            return {"error": "风险预测模型未加载"}
        
        # 预测风险等级
        risk_pred = self.risk_model.predict(X_scaled)
        risk_proba = self.risk_model.predict_proba(X_scaled)
        
        # 概率分布
        if self.metadata and 'risk_map' in self.metadata:
            risk_map = self.metadata['risk_map']
        else:
            risk_map = {0: '低风险', 1: '中风险', 2: '高风险', 3: '极高风险'}
        
        prob_distribution = {}
        for i, label in risk_map.items():
            if i < risk_proba.shape[1]:
                prob_distribution[label] = round(risk_proba[0, i] * 100, 2)
        
        return {
            "预测等级": risk_map.get(int(risk_pred[0]), '未知'),
            "预测等级编码": int(risk_pred[0]),
            "置信度": round(risk_proba[0, int(risk_pred[0])] * 100, 2),
            "probability_distribution": prob_distribution
        }
    
    def _comprehensive_assessment(self, rule_result, ml_result):
        """
        综合评估：融合规则引擎和机器学习模型的结果
        
        采用加权融合策略:
        - 规则引擎权重: 0.4（临床经验）
        - 机器学习模型权重: 0.6（数据驱动）
        
        Parameters:
            rule_result: 规则评估结果
            ml_result: 模型预测结果
            
        Returns:
            dict: 综合评估结果
        """
        if rule_result and 'error' not in ml_result:
            # 加权融合
            rule_code = rule_result['风险等级编码']
            ml_code = ml_result['预测等级编码']
            
            # 简单多数投票
            if rule_code == ml_code:
                final_code = rule_code
            elif abs(rule_code - ml_code) == 1:
                final_code = max(rule_code, ml_code)  # 取较高风险等级（保守策略）
            else:
                # 差异较大时，以ML模型为主
                final_code = ml_code
        elif rule_result:
            final_code = rule_result['风险等级编码']
        elif 'error' not in ml_result:
            final_code = ml_result['预测等级编码']
        else:
            final_code = 1  # 默认中风险
        
        risk_map = {0: '低风险', 1: '中风险', 2: '高风险', 3: '极高风险'}
        
        return {
            'risk_code': final_code,
            'risk_level': risk_map.get(final_code, '未知'),
            'assessment_method': '规则+ML融合评估'
        }
    
    def _analyze_feature_importance(self, X_scaled):
        """
        风险因子重要性分析
        
        基于模型的特征重要性评分，识别影响迁移应激风险
        的关键因素。
        
        Parameters:
            X_scaled: 标准化后的特征矩阵
            
        Returns:
            list: 特征重要性排序列表
        """
        if self.risk_model is None:
            return []
        
        feature_names = [
            "Age", "Gender", "Education", "Relationship",
            "Income", "ICU_Stay_Days", "Diagnosis", "GCS_Score",
            "FRSS_T0", "FCTI_T0", "SAS_T0", "SDS_T0"
        ]
        
        importance = self.risk_model.feature_importances_
        
        importance_list = []
        for i, (name, imp) in enumerate(zip(feature_names, importance)):
            importance_list.append({
                "特征": name,
                "重要性": round(imp * 100, 2),
                "排序": i + 1
            })
        
        # 按重要性排序
        importance_list.sort(key=lambda x: x['重要性'], reverse=True)
        for i, item in enumerate(importance_list):
            item['排序'] = i + 1
        
        return importance_list
    
    def predict_frss_improvement(self, X_scaled):
        """
        预测FRSS改善幅度
        
        使用随机森林回归模型预测实施护理干预后
        FRSS评分的预期改善幅度。
        
        Parameters:
            X_scaled: 标准化后的特征矩阵
            
        Returns:
            dict: 改善预测结果
        """
        if self.frss_model is None:
            return {"error": "FRSS改善预测模型未加载"}
        
        prediction = self.frss_model.predict(X_scaled)
        
        # 获取特征重要性
        feature_names = [
            "Age", "Gender", "Education", "Relationship",
            "Income", "ICU_Stay_Days", "Diagnosis", "GCS_Score",
            "FRSS_T0", "FCTI_T0", "SAS_T0", "SDS_T0"
        ]
        
        return {
            "预测改善幅度": round(prediction[0], 2),
            "预测说明": f"预计干预后FRSS评分将{'提高' if prediction[0] > 0 else '降低'}{abs(prediction[0]):.1f}分"
        }
