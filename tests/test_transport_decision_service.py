"""
运输决策支持服务测试模块
"""
import unittest
from datetime import datetime, timedelta
from src.services.transport_decision_service import TransportDecisionService

class TestTransportDecisionService(unittest.TestCase):
    """测试运输决策支持服务"""
    
    def setUp(self):
        """测试前准备"""
        self.service = TransportDecisionService()
        
        # 创建测试数据
        self.test_artwork_data = {
            'artwork_value': 100000.0,
            'fragility_score': 4,
            'days_to_deadline': 14,
            'is_international': True,
            'chargeable_weight': 500.0,
            'requires_climate_control': True,
            'requires_custom_crating': True,
            'requires_art_handler': False,
            'deadline': datetime.now() + timedelta(days=14),
            'Length': 100,
            'Width': 50,
            'Height': 30,
            'Actual_Weight': 100,
            'Origin_Country': 'CN',
            'Destination_Country': 'US',
            'Artwork_Type': 'painting'
        }
        
        self.test_distance = 8000.0  # 公里
    
    def test_calculate_risk_score(self):
        """测试风险评分计算"""
        risk_result = self.service.calculate_risk_score(
            artwork_value=self.test_artwork_data['artwork_value'],
            fragility_score=self.test_artwork_data['fragility_score'],
            days_to_deadline=self.test_artwork_data['days_to_deadline'],
            distance_km=self.test_distance,
            special_handling_count=2
        )
        
        # 检查结果结构
        self.assertIn('total_score', risk_result)
        self.assertIn('components', risk_result)
        self.assertIn('risk_level', risk_result)
        
        # 检查评分范围
        self.assertTrue(0 <= risk_result['total_score'] <= 1)
        
        # 检查风险等级
        self.assertIn(risk_result['risk_level'], ['低风险', '中等风险', '高风险'])
    
    def test_recommend_transport_mode(self):
        """测试运输模式推荐"""
        recommendation = self.service.recommend_transport_mode(
            is_international=self.test_artwork_data['is_international'],
            artwork_value=self.test_artwork_data['artwork_value'],
            weight=self.test_artwork_data['chargeable_weight'],
            days_to_deadline=self.test_artwork_data['days_to_deadline'],
            distance_km=self.test_distance
        )
        
        # 检查结果结构
        self.assertIn('recommended_mode', recommendation)
        self.assertIn('reasons', recommendation)
        
        # 检查推荐模式
        self.assertIn(recommendation['recommended_mode'], 
                     ['国际空运', '国内空运', '国内陆运'])
        
        # 检查原因列表
        self.assertTrue(len(recommendation['reasons']) > 0)
    
    def test_generate_time_plan(self):
        """测试时间规划生成"""
        time_plan = self.service.generate_time_plan(
            transport_mode='国际空运',
            distance_km=self.test_distance,
            requires_special_handling=True,
            deadline=self.test_artwork_data['deadline']
        )
        
        # 检查结果结构
        self.assertIn('total_estimated_days', time_plan)
        self.assertIn('breakdown', time_plan)
        self.assertIn('recommended_start_date', time_plan)
        self.assertIn('latest_start_date', time_plan)
        
        # 检查时间估算的合理性
        self.assertTrue(time_plan['total_estimated_days'] > 0)
        self.assertTrue(time_plan['breakdown']['transport_days'] > 0)
        
        # 检查日期逻辑
        self.assertTrue(time_plan['recommended_start_date'] <= time_plan['latest_start_date'])
        self.assertTrue(time_plan['latest_start_date'] < time_plan['deadline'])
    
    def test_estimate_total_cost(self):
        """测试总成本估算"""
        cost_result = self.service.estimate_total_cost(
            base_price=10000.0,
            artwork_value=self.test_artwork_data['artwork_value'],
            special_handling_cost=2500.0,
            is_express=True
        )
        
        # 检查结果结构
        self.assertIn('total_cost', cost_result)
        self.assertIn('breakdown', cost_result)
        self.assertIn('insurance_details', cost_result)
        
        # 检查成本计算的合理性
        self.assertTrue(cost_result['total_cost'] > cost_result['breakdown']['base_price'])
        self.assertTrue(cost_result['breakdown']['insurance_cost'] > 0)
        
        # 检查快递附加费
        self.assertEqual(
            cost_result['breakdown']['express_surcharge'],
            cost_result['breakdown']['base_price'] * 0.3
        )
    
    def test_generate_full_recommendation(self):
        """测试完整运输建议生成"""
        recommendation = self.service.generate_full_recommendation(
            artwork_data=self.test_artwork_data,
            distance_km=self.test_distance
        )
        
        # 检查结果结构
        self.assertIn('risk_assessment', recommendation)
        self.assertIn('transport_recommendation', recommendation)
        self.assertIn('time_plan', recommendation)
        self.assertIn('cost_estimation', recommendation)
        self.assertIn('price_prediction_details', recommendation)
        
        # 检查风险评估
        self.assertIn('total_score', recommendation['risk_assessment'])
        self.assertIn('risk_level', recommendation['risk_assessment'])
        
        # 检查运输建议
        self.assertIn('recommended_mode', recommendation['transport_recommendation'])
        self.assertTrue(len(recommendation['transport_recommendation']['reasons']) > 0)
        
        # 检查时间规划
        self.assertIn('total_estimated_days', recommendation['time_plan'])
        self.assertIn('recommended_start_date', recommendation['time_plan'])
        
        # 检查成本估算
        self.assertIn('total_cost', recommendation['cost_estimation'])
        self.assertIn('breakdown', recommendation['cost_estimation'])
        
        # 检查价格预测
        self.assertIn('base_price', recommendation['price_prediction_details'])
        self.assertIn('confidence_interval', recommendation['price_prediction_details'])

if __name__ == '__main__':
    unittest.main() 