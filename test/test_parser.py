import os
import sys
import json
# 当前目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hengline.agent.script_parser_agent import ScriptParserAgent

# 测试剧本文本（基于用户提供的示例）
test_script = '''场景：城市公寓客厅，深夜11点
林然裹着毯子坐在沙发上，电视静音播放着老电影。她的手机突然震动，屏幕亮起显示“未知号码”。她犹豫了一下，接起电话，听到对面传来熟悉的声音："是我"。她的手微微发抖，轻声问道："陈默？你还好吗？"对方沉默了几秒，说："我回来了"。'''

def test_parser():
    """测试剧本解析器的修改效果"""
    print("开始测试剧本解析器...")
    
    # 初始化解析器
    parser = ScriptParserAgent()
    
    # 解析剧本
    print("\n解析剧本中...")
    result = parser.parse_script(test_script)
    
    # 打印结果
    print("\n解析结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 验证修改是否解决了问题
    print("\n验证修改效果:")
    
    # 检查1: 动作是否被拆分
    print("\n1. 检查动作拆分:")
    if result and "scenes" in result and len(result["scenes"]) > 0:
        actions = result["scenes"][0].get("actions", [])
        print(f"   拆分后的动作数量: {len(actions)}")
        for i, action in enumerate(actions):
            if "action" in action:
                emotion = action.get('emotion', '无')
                print(f"   动作 {i+1}: {action['character']} - {action['action']} (情绪: {emotion})")
            elif "dialogue" in action:
                emotion = action.get('emotion', '无')
                print(f"   对话 {i+1}: {action['character']} - {action['dialogue']} (情绪: {emotion})")
    
    # 检查2: 角色行为耦合
    print("\n2. 检查角色行为耦合:")
    has_phone_character = False
    for action in actions:
        if action.get("character") == "陈默":
            has_phone_character = True
            print(f"   警告: 发现独立的陈默动作: {action}")
    if not has_phone_character:
        print("   ✓ 通过: 没有发现独立的陈默动作，电话对话已作为林然动作的一部分")
    
    # 检查3: 场景信息冗余
    print("\n3. 检查场景信息冗余:")
    has_redundancy = False
    for action in actions:
        action_text = action.get("action", "")
        if "城市公寓内" in action_text or "客厅内" in action_text:
            has_redundancy = True
            print(f"   警告: 动作中仍有场景冗余信息: {action_text}")
    if not has_redundancy:
        print("   ✓ 通过: 动作描述中没有场景冗余信息")
    
    # 检查4: 情绪描述
    print("\n4. 检查情绪描述:")
    for action in actions:
        emotion = action.get("emotion", "无")
        if len(emotion) > 8:
            print(f"   警告: 情绪描述可能过于文学化: {emotion}")
        else:
            print(f"   ✓ 简洁情绪标签: {emotion}")
    
    # 检查5: 场景结构
    print("\n5. 检查场景结构:")
    if result and "scenes" in result and len(result["scenes"]) > 0:
        scene = result["scenes"][0]
        print(f"   场景位置: {scene.get('location')}")
        print(f"   场景时间: {scene.get('time_of_day')}")
        print(f"   场景氛围: {scene.get('atmosphere')}")
        print(f"   角色数量: {len(scene.get('characters', []))}")
        for char in scene.get('characters', []):
            print(f"     角色: {char.get('name')}")
            # 检查是否存在actions字段及其是否为空（避免重复）
            has_actions = 'actions' in char and len(char.get('actions', [])) > 0
            print(f"     角色是否包含actions字段: {'✓ 包含' if 'actions' in char else '✗ 不包含'}")
            print(f"     actions字段是否为空: {'✓ 为空' if 'actions' in char and len(char.get('actions', [])) == 0 else '✗ 不为空'}")
    
    # 检查6: 动作的独特性和具体性
    print("\n6. 检查动作的独特性和具体性:")
    action_texts = []
    for action in actions:
        action_text = action.get('action', '')
        action_texts.append(action_text)
        print(f"   动作文本: '{action_text}'")
        print(f"     长度: {len(action_text)} 字符")
        print(f"     是否包含具体身体动作: {'✓ 是' if any(keyword in action_text for keyword in ['手指', '双手', '身体', '坐姿', '前倾', '转身']) else '✗ 否'}")
    
    # 检查动作是否重复
    if len(action_texts) != len(set(action_texts)):
        print("   ✗ 警告: 发现重复的动作描述")
    else:
        print("   ✓ 通过: 所有动作描述都是唯一的")

if __name__ == "__main__":
    test_parser()