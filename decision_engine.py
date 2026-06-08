#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 decision_engine.py - 智能护理决策引擎模块
 神经重症患者迁移应激护理决策系统 V1.0
 
 功能说明:
   1. 基于梯度提升分类器的护理决策推荐
   2. 个性化护理方案生成
   3. T0-T3四阶段护理计划制定
   4. 护理措施优先级排序
   5. 决策依据和循证支持
   6. 多方案对比与最优选择
================================================================================
"""
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_DIR, CARE_DECISIONS


class CareDecisionEngine:
    """
    智能护理决策引擎
    
    基于患者的风险评估结果和多维特征数据，运用机器学习模型
    推荐最优的护理决策方案，并提供详细的护理措施、实施计划
    和循证支持。
    
    Attributes:
        decision_model: 梯度提升分类器，用于护理决策分类
        scaler: 数据标准化器
        metadata: 模型元数据
    """
    
    def __init__(self, model_dir=MODEL_DIR):
        """
        初始化决策引擎
        
        Parameters:
            model_dir: 模型文件存储目录
        """
        self.model_dir = model_dir
        self.decision_model = None
        self.scaler = None
        self.metadata = None
        self._load_models()
    
    def _load_models(self):
        """加载预训练的决策模型"""
        decision_model_path = os.path.join(self.model_dir, "decision_model.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        metadata_path = os.path.join(self.model_dir, "metadata.pkl")
        
        if os.path.exists(decision_model_path):
            with open(decision_model_path, 'rb') as f:
                self.decision_model = pickle.load(f)
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
    
    def generate_decision(self, X_scaled, patient_info=None, risk_report=None):
        """
        生成智能护理决策
        
        决策流程:
        1. 使用ML模型预测最优护理决策
        2. 结合风险等级进行方案调整
        3. 生成个性化护理方案
        4. 制定T0-T3四阶段实施计划
        5. 提供决策依据和循证支持
        
        Parameters:
            X_scaled: 标准化后的特征向量
            patient_info: 患者信息字典
            risk_report: 风险评估报告
            
        Returns:
            dict: 护理决策报告
        """
        # 1. 模型预测
        ml_decision = self._ml_decision(X_scaled)
        
        # 2. 基于风险的规则调整
        rule_adjustment = self._rule_based_adjustment(patient_info, risk_report)
        
        # 3. 综合决策
        final_decision = self._integrate_decisions(ml_decision, rule_adjustment)
        
        # 4. 生成护理方案
        care_plan = self._generate_care_plan(final_decision, patient_info, risk_report)
        
        # 5. 生成T0-T3实施计划
        timeline = self._generate_timeline(final_decision, care_plan)
        
        # 6. 构建完整报告
        report = {
            "推荐决策": final_decision,
            "决策置信度": ml_decision.get('confidence', 0),
            "决策依据": self._generate_rationale(final_decision, risk_report),
            "护理方案": care_plan,
            "实施计划": timeline,
            "替代方案": ml_decision.get('alternatives', []),
            "ML预测详情": ml_decision,
            "规则调整": rule_adjustment
        }
        
        return report
    
    def _ml_decision(self, X_scaled):
        """
        基于机器学习的决策推荐
        
        Parameters:
            X_scaled: 标准化后的特征向量
            
        Returns:
            dict: ML预测结果
        """
        if self.decision_model is None:
            return {"error": "决策模型未加载", "decision_key": "standard_care"}
        
        # 预测决策类别
        decision_pred = self.decision_model.predict(X_scaled)
        decision_proba = self.decision_model.predict_proba(X_scaled)
        
        # 决策标签映射
        if self.metadata and 'decision_map' in self.metadata:
            decision_map = self.metadata['decision_map']
        else:
            decision_map = {
                0: 'intensive_care',
                1: 'enhanced_support',
                2: 'maintenance',
                3: 'standard_care'
            }
        
        pred_code = int(decision_pred[0])
        decision_key = decision_map.get(pred_code, 'standard_care')
        confidence = round(decision_proba[0, pred_code] * 100, 2)
        
        # 替代方案（概率次高的决策）
        alternatives = []
        sorted_indices = np.argsort(decision_proba[0])[::-1]
        for idx in sorted_indices[1:3]:  # 取Top2替代方案
            alt_key = decision_map.get(idx, '')
            if alt_key and alt_key != decision_key:
                alternatives.append({
                    "方案": CARE_DECISIONS.get(alt_key, {}).get('name', alt_key),
                    "概率": round(decision_proba[0, idx] * 100, 2)
                })
        
        return {
            "decision_key": decision_key.strip(),
            "confidence": confidence,
            "probability_distribution": {
                CARE_DECISIONS.get(decision_map.get(i, ''), {}).get('name', f'方案{i}'): 
                round(decision_proba[0, i] * 100, 2) 
                for i in range(len(decision_map)) if i < decision_proba.shape[1]
            },
            "alternatives": alternatives
        }
    
    def _rule_based_adjustment(self, patient_info, risk_report):
        """
        基于规则的决策调整
        
        根据风险等级和临床规则对ML推荐的决策进行调整:
        - 极高风险 → 强制升级为强化护理
        - 存在焦虑+抑郁 → 增强心理支持
        - GCS重度昏迷 → 增加社工介入
        
        Parameters:
            patient_info: 患者信息
            risk_report: 风险评估报告
            
        Returns:
            dict: 规则调整建议
        """
        adjustments = []
        
        if risk_report and '综合评估' in risk_report:
            risk_code = risk_report['综合评估'].get('risk_code', 1)
            
            if risk_code == 3:  # 极高风险
                adjustments.append({
                    "调整类型": "强制升级",
                    "说明": "风险等级为极高风险，自动升级为强化护理干预",
                    "优先级": "紧急"
                })
            elif risk_code == 2:  # 高风险
                adjustments.append({
                    "调整类型": "建议升级",
                    "说明": "风险等级为高风险，建议采用增强支持护理",
                    "优先级": "高"
                })
        
        if patient_info:
            # 焦虑+抑郁双重症状
            sas = patient_info.get('SAS_T0', 0)
            sds = patient_info.get('SDS_T0', 0)
            if sas >= 50 and sds >= 53:
                adjustments.append({
                    "调整类型": "增加心理支持",
                    "说明": "家属同时存在焦虑和抑郁症状，需增加心理支持和危机干预",
                    "优先级": "高"
                })
            
            # 长期ICU住院
            icu_days = patient_info.get('ICU_Stay_Days', 0)
            if icu_days > 14:
                adjustments.append({
                    "调整类型": "增加分离焦虑关注",
                    "说明": f"ICU住院{icu_days}天，需重点关注分离焦虑和ICU依赖",
                    "优先级": "中"
                })
        
        return {
            "调整建议": adjustments,
            "是否需要调整": len(adjustments) > 0
        }
    
    def _integrate_decisions(self, ml_decision, rule_adjustment):
        """
        融合ML预测和规则调整的最终决策
        
        Parameters:
            ml_decision: ML模型预测结果
            rule_adjustment: 规则调整建议
            
        Returns:
            dict: 最终决策
        """
        base_key = ml_decision.get('decision_key', 'standard_care').strip()
        
        # 检查是否有强制升级
        if rule_adjustment and '调整建议' in rule_adjustment:
            for adj in rule_adjustment['调整建议']:
                if adj['调整类型'] == '强制升级':
                    base_key = 'intensive_care'
                    break
        
        # 确保key有效
        if base_key not in CARE_DECISIONS:
            base_key = 'standard_care'
        
        decision_info = CARE_DECISIONS[base_key]
        
        return {
            "decision_key": base_key,
            "方案名称": decision_info['name'],
            "方案描述": decision_info['description'],
            "优先级": decision_info['priority'],
            "颜色标识": decision_info['color']
        }
    
    def _generate_care_plan(self, final_decision, patient_info, risk_report):
        """
        生成详细护理方案
        
        Parameters:
            final_decision: 最终决策
            patient_info: 患者信息
            risk_report: 风险评估报告
            
        Returns:
            dict: 护理方案详情
        """
        decision_key = final_decision['decision_key']
        
        if decision_key not in CARE_DECISIONS:
            decision_key = 'standard_care'
        
        base_plan = CARE_DECISIONS[decision_key]
        
        # 个性化调整
        personalized_measures = base_plan['measures'].copy()
        
        if patient_info:
            # 根据诊断类型添加特异性措施
            diagnosis = patient_info.get('Diagnosis', '')
            if '脑出血' in diagnosis or '蛛网膜下腔出血' in diagnosis:
                personalized_measures.append(
                    "特别关注出血后脑水肿和再出血风险的宣教"
                )
            elif '脑梗死' in diagnosis:
                personalized_measures.append(
                    "重点关注溶栓/取栓后的护理要点和康复时机"
                )
            elif '颅脑损伤' in diagnosis:
                personalized_measures.append(
                    "加强创伤后认知障碍和情绪障碍的评估与干预"
                )
        
        return {
            "方案名称": base_plan['name'],
            "方案描述": base_plan['description'],
            "优先级": base_plan['priority'],
            "护理措施": personalized_measures,
            "预期目标": self._set_expected_goals(decision_key),
            "评估指标": self._set_evaluation_metrics(decision_key)
        }
    
    def _set_expected_goals(self, decision_key):
        """
        设置预期护理目标
        
        Parameters:
            decision_key: 决策类型标识
            
        Returns:
            list: 预期目标列表
        """
        goals_map = {
            'intensive_care': [
                "FRSS评分提高≥15分",
                "SAS评分降低≥10分",
                "SDS评分降低≥10分",
                "FCTI评分降低≥8分",
                "家属满意度≥90分"
            ],
            'enhanced_support': [
                "FRSS评分提高≥10分",
                "SAS评分降低≥8分",
                "SDS评分降低≥8分",
                "FCTI评分降低≥5分",
                "家属满意度≥85分"
            ],
            'standard_care': [
                "FRSS评分提高≥5分",
                "SAS评分降低≥5分",
                "SDS评分降低≥5分",
                "FCTI评分降低≥3分",
                "家属满意度≥80分"
            ],
            'maintenance': [
                "FRSS评分保持稳定或小幅提高",
                "SAS/SDS评分无显著恶化",
                "家属满意度≥75分"
            ]
        }
        return goals_map.get(decision_key, goals_map['standard_care'])
    
    def _set_evaluation_metrics(self, decision_key):
        """
        设置评估指标
        
        Parameters:
            decision_key: 决策类型标识
            
        Returns:
            list: 评估指标列表
        """
        base_metrics = [
            "FRSS迁移应激量表评分",
            "FCTI照顾能力量表评分",
            "SAS焦虑自评量表评分",
            "SDS抑郁自评量表评分",
            "协同服务满意度评分"
        ]
        
        if decision_key in ['intensive_care', 'enhanced_support']:
            base_metrics.extend([
                "社工转介有效率",
                "资源链接成功率",
                "家庭会议出席率",
                "干预保真度评分"
            ])
        
        return base_metrics
    
    def _generate_timeline(self, final_decision, care_plan):
        """
        生成T0-T3四阶段实施计划
        
        Parameters:
            final_decision: 最终决策
            care_plan: 护理方案
            
        Returns:
            dict: 时间线计划
        """
        decision_key = final_decision['decision_key']
        
        timeline = {
            "T0_ICU入院24h内": {
                "护士任务": [
                    "完成FRSS量表初筛（14条目）",
                    "ICU入院宣教（探视制度、治疗计划、预后说明）",
                    "使用叙事护理技术建立信任关系",
                    "神经重症特异性评估（意识障碍沟通、预后不确定性、长期照护准备度）"
                ],
                "社工任务": [
                    "FRSS≥55分者24h内完成深度心理社会评估",
                    "简化版家庭功能评估（家庭结构、照顾者、社会支持、经济状况）",
                    "情绪支持和危机干预",
                    "经济负担初筛"
                ] if decision_key in ['intensive_care', 'enhanced_support'] else [
                    "待命，根据护士转介启动评估"
                ],
                "协同要点": [
                    "护士完成初筛后立即转介高风险对象",
                    "社工24h内完成评估并反馈",
                    "共同识别高风险家庭"
                ]
            },
            "T1_转出前3天": {
                "护士任务": [
                    "联合查房评估转出准备度",
                    "个性化转出准备计划制定",
                    "照护技能培训（气道管理、营养支持、康复训练）",
                    "转出流程详细说明"
                ],
                "社工任务": [
                    "结构化家庭会议（5步法）",
                    "FCTI/SAS/SDS量表评估",
                    "疾病适应辅导和认知重构",
                    "经济负担评估和资源链接启动"
                ] if decision_key in ['intensive_care', 'enhanced_support'] else [
                    "按需开展家庭会议"
                ],
                "协同要点": [
                    "至少一次联合查房",
                    "家庭会议由社工主导、护士提供医学支持",
                    "共同完成《转出准备计划》"
                ]
            },
            "T2_转出当天": {
                "护士任务": [
                    "执行标准化转出流程（SOP）",
                    "床旁交接（病情、护理要点、康复计划）",
                    "转出后注意事项说明"
                ],
                "社工任务": [
                    "转出支持家庭会议（缓解分离焦虑）",
                    "照护分工协调（《家庭照护分工表》）",
                    "情绪支持和资源链接准备"
                ] if decision_key in ['intensive_care', 'enhanced_support'] else [
                    "电话随访确认转出适应情况"
                ],
                "协同要点": [
                    "同步行动，确保医学安全和情感支持同步",
                    "社工在家庭会议中强调护士交接的照护重点",
                    "共同完成《转出交接记录》"
                ]
            },
            "T3_转出后1天": {
                "护士任务": [
                    "共同随访（电话/床旁）评估适应情况",
                    "FCTI量表再评估",
                    "补充照护指导",
                    "结局数据采集（FRSS/FCTI/SAS/SDS T3测评）"
                ],
                "社工任务": [
                    "心理适应和家庭功能评估",
                    "满意度调查",
                    "正式资源链接（社区康复、经济援助、心理支持）",
                    "个案总结和档案归档"
                ] if decision_key in ['intensive_care', 'enhanced_support'] else [
                    "满意度调查和资源信息提供"
                ],
                "协同要点": [
                    "密切配合完成所有量表测评",
                    "随访结果在月度例会上汇报",
                    "高风险家庭启动长期随访机制"
                ]
            }
        }
        
        return timeline
    
    def _generate_rationale(self, final_decision, risk_report):
        """
        生成决策依据说明
        
        Parameters:
            final_decision: 最终决策
            risk_report: 风险评估报告
            
        Returns:
            str: 决策依据说明
        """
        risk_level = risk_report.get('风险等级', '中风险') if risk_report else '中风险'
        decision_name = final_decision['方案名称']
        
        rationale = f"基于风险评估结果为'{risk_level}'，系统推荐采用'{decision_name}'方案。"
        
        if risk_report and '风险因子重要性' in risk_report:
            top_factors = risk_report['风险因子重要性'][:3]
            factor_names = [f['特征'] for f in top_factors]
            rationale += f"主要风险因子包括：{', '.join(factor_names)}。"
        
        rationale += "该决策综合了机器学习模型预测和临床规则引擎的评估结果。"
        
        return rationale
