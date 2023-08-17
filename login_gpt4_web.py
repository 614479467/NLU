from chatgpt_wrapper.main_browser_wrapper import *

def login():

    shell = wrapper_init('reinstall')
    shell.cmdloop()


if __name__ == '__main__':
    '''
    1、首先呢先pip install -r requirements.txt，我使用的是一个python38的环境
    2、其次呢，需要 playwright install firefox 安装一个驱动
    3、测试demo吧
    
    '''
    login() # 第一次使用需要先用这个登录一下，会弹出一个浏览器要求登录openAI账号，手动登录后即可。

