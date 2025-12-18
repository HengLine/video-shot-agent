import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hengline.agent.script_parser_agent import ScriptParserAgent

def test_scene_parser():
    # 原剧本内容
    script_text = "深夜11点，城市公寓内，窗外下着大雨。林然裹着毯子坐在沙发上，电视静音播放着老电影。手机突然震动，屏幕亮起\"未知号码\"。她犹豫了一下，还是接了起来。电话那头沉默了几秒，传来一个熟悉又陌生的男声：\"是我。\"林然的手微微发抖，轻声问：\"……陈默？你还好吗？\"对方停顿片刻，低声说：\"我回来了。\""
    
    # 创建解析器实例
    parser_agent = ScriptParserAgent()
    
    # 解析剧本
    result = parser_agent.parse_script(script_text)
    
    # 打印结果
    print("解析结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 检查场景信息
    if result and "scenes" in result and result["scenes"]:
        scene = result["scenes"][0]
        print("\n提取的场景信息:")
        print(f"地点: {scene.get('location', '未识别')}")
        print(f"时间(time字段): {scene.get('time', '未识别')}")
        print(f"时间(time_of_day字段): {scene.get('time_of_day', '未识别')}")
        
        # 验证修改是否成功
        location_correct = "公寓" in scene.get('location', '')
        time_correct = "深夜" in scene.get('time', '') or "深夜" in scene.get('time_of_day', '')
        
        if location_correct and time_correct:
            print("\n测试成功! 正确识别了'城市公寓'和'深夜11点'。")
        else:
            print("\n测试失败! 未能完全正确识别场景信息。")
            if not location_correct:
                print("  - 地点识别失败")
            if not time_correct:
                print("  - 时间识别失败")
    else:
        print("\n测试失败! 未能提取场景信息。")

if __name__ == "__main__":
    import json
    test_scene_parser()