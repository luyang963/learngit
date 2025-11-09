import requests
import json
import time
import random
import os
import sys

# ==================== 关键修改：使用相对路径 ====================
# 计算WebShop相对路径
current_dir = os.path.dirname(__file__)  # ragen/ 目录
project_root = os.path.dirname(current_dir)  # RAGEN_MODAL/ 目录
webshop_path = os.path.join(project_root, 'WebShop')

if webshop_path not in sys.path:
    sys.path.insert(0, webshop_path)
    print(f"🔧 添加WebShop路径: {webshop_path}")

try:
    from webshop import WebShopEnv as OfficialWebShopEnv
    WEBSHOP_AVAILABLE = True
    print("✅ 成功导入本地WebShop环境")
except ImportError as e:
    WEBSHOP_AVAILABLE = False
    print(f"❌ 导入本地WebShop失败: {e}")
    print("🔧 使用模拟模式")

class WebShopEnv:
    def __init__(self, server_url="http://localhost:3000", max_steps=15):
        self.server_url = server_url
        self.max_steps = max_steps
        self.current_step = 0
        self.session_id = None
        
        # 关键修改：检查是否使用真实WebShop环境
        self.use_real_webshop = WEBSHOP_AVAILABLE and os.environ.get("USE_REAL_WEBSHOP", "true").lower() == "true"
        
        if self.use_real_webshop:
            print("🎯 使用真实WebShop环境")
            # 初始化真实WebShop环境
            self._init_real_webshop()
        else:
            print("🔧 使用WebShop模拟模式")
            # 初始化模拟数据
            self._init_simulation()
    
    def _init_real_webshop(self):
        """初始化真实WebShop环境"""
        try:
            self.real_env = OfficialWebShopEnv()
            print("✅ 真实WebShop环境初始化成功")
        except Exception as e:
            print(f"❌ 真实WebShop环境初始化失败: {e}")
            print("🔄 切换到模拟模式")
            self.use_real_webshop = False
            self._init_simulation()
    
    def _init_simulation(self):
        """初始化模拟数据"""
        self.tasks = [
            "Find and buy a red shirt",
            "Purchase a classic blanket", 
            "Buy a wireless mouse with good ratings",
            "Find a laptop under $1000",
            "Get a blue jeans in size 32",
            "Purchase a wireless keyboard",
            "Find a black backpack with laptop compartment",
            "Buy a stainless steel water bottle"
        ]
        
        self.simulated_products = {
            'shirt': [{'id': 1, 'name': 'Red Cotton Shirt', 'color': 'red', 'price': 29.99}],
            'blanket': [{'id': 3, 'name': 'Classic Wool Blanket', 'type': 'classic', 'price': 49.99}],
            'jeans': [{'id': 5, 'name': 'Blue Denim Jeans Size 32', 'color': 'blue', 'size': 32, 'price': 59.99}],
            'laptop': [{'id': 7, 'name': 'Gaming Laptop $999', 'price': 999.99}],
            'mouse': [{'id': 9, 'name': 'Wireless Gaming Mouse', 'type': 'wireless', 'rating': 4.5, 'price': 49.99}]
        }
    
    def reset(self, instruction=None):
        """重置环境"""
        self.current_step = 0
        
        if instruction is None:
            instruction = random.choice(self.tasks) if not self.use_real_webshop else "Find a product"
        
        self.current_instruction = instruction
        
        if self.use_real_webshop:
            try:
                # 使用真实WebShop环境
                observation = self.real_env.reset()
                self.session_id = f"real_webshop_{int(time.time())}"
                print(f"🎯 真实WebShop任务开始: {instruction}")
                return observation, {'session_id': self.session_id, 'instruction': instruction, 'real_environment': True}
                
            except Exception as e:
                print(f"❌ 真实WebShop reset失败: {e}")
                print("🔄 切换到模拟模式")
                self.use_real_webshop = False
        
        # 模拟模式
        self.session_id = f"sim_{int(time.time())}"
        observation = f"欢迎！请{instruction}\n页面显示搜索框和商品分类。"
        
        print(f"🎯 模拟环境任务开始: {instruction}")
        return observation, {'session_id': self.session_id, 'instruction': instruction, 'real_environment': False}
    
    def step(self, action, session_id=None):
        """执行动作"""
        if session_id is None:
            session_id = self.session_id
            
        self.current_step += 1
        
        if self.use_real_webshop:
            try:
                # 使用真实WebShop环境
                observation, reward, done, info = self.real_env.step(action)
                
                # 确保返回格式一致
                if info is None:
                    info = {}
                info.update({
                    'session_id': session_id,
                    'step': self.current_step,
                    'action': action,
                    'real_environment': True
                })
                
                return observation, reward, done, info
                
            except Exception as e:
                print(f"❌ 真实WebShop step失败: {e}")
                self.use_real_webshop = False
        
        # 模拟模式
        observation, reward, done = self._simulate_step(action)
        
        info = {
            'session_id': session_id,
            'step': self.current_step,
            'action': action,
            'real_environment': False
        }
        
        return observation, reward, done, info
    
    def _simulate_step(self, action):
        """模拟环境步骤"""
        action_type = action.split('[')[0] if '[' in action else action
        
        if action_type == "search":
            reward = 0.2
            done = False
            observation = f"搜索结果页面 - 显示相关商品列表"
                
        elif action_type == "click":
            reward = 0.3
            done = False
            observation = f"商品详情页面 - 显示商品信息"
                
        elif action_type == "buy":
            success_prob = 0.6  # 基础成功率
            if random.random() < success_prob:
                reward = 1.0
                done = True
                observation = "🎉 购买成功！任务完成！"
            else:
                reward = 0.1
                done = False
                observation = "⚠️ 购买失败，请检查商品或重试"
                
        else:
            reward = -0.1
            done = False
            observation = "❌ 无效动作格式"
        
        # 步数限制
        if self.current_step >= self.max_steps and not done:
            done = True
            reward = 0.0
            observation = "⏰ 步数限制达到，任务失败"
        
        return observation, reward, done
    
    def close(self):
        """关闭环境"""
        if self.use_real_webshop:
            try:
                self.real_env.close()
                print("✅ 真实WebShop环境关闭成功")
            except Exception as e:
                print(f"⚠️ 真实WebShop环境关闭失败: {e}")