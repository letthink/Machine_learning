# coding=utf-8
# Author: Lishiyao
# CreatTime: 2021/2/26 20 55
# FileName: day1 
# Description: Simple introduction of the code
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba

def datasets_demo():
    """
    sklearn数据集使用
    """
    iris = load_iris()
    print("鸢尾花数据集\n", iris)
    print("DESCR\n", iris.DESCR)
    print("name\n", iris["feature_names"])
    print("data\n", iris.data)
    print("shape\n", iris.data.shape)

    # 数据的划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("x_train训练集的特征值：\n", x_train)
    print("x_test:测试集的特征值\n", x_test)
    print("y_train训练集的目标值\n", y_train)
    print("y_train测试集的目标值\n", y_test)

def dict_demo():
    # 字典特征抽取
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    # 1.实例化一个转换器类
    transfer = DictVectorizer(sparse=False)
    # 2.调用fit_transform
    data = transfer.fit_transform(data)
    print("返回的结果:\n",data)
    #打印特征名字
    print("特征名字:\n",transfer.get_feature_names())
    return None

def text_count_demo():
    #对文本进行特征抽取，countvetorizer
    data = ["life is short,i like python and like C too", "life is too long,i dislike python"]
    #1.实例化一个转换器类
    transfer = CountVectorizer()
    #2.调用fit_transform
    data = transfer.fit_transform(data)
    print("文本特征抽取结果\n",data.toarray()) #实例的一种格式转换 toarray()
    print("返回特征名字\n",transfer.get_feature_names())

def cut_word(text):
    return " ".join(jieba.lcut(text))
    # return " ".join(list(jieba.cut(text)))

def text_chinese_count_demo():
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    text_list = []
    for i in data:
        text_list.append(cut_word(i))
    #1.实例化转化器类
    transfer = CountVectorizer()
    # print(text_list,"text.list测试")
    #2.调用fit_transform
    data = transfer.fit_transform(text_list) #注意变量的选择 data是还未处理过的数据
    print("特征值：\n",data.toarray())
    print("名称：\n",transfer.get_feature_names())


if __name__ == '__main__':
    print("测试开始")
    #第一部分：sklearn数据集的使用
    # datasets_demo()
    #第二部分：字典的特征抽取
    # dict_demo()
    #第三部分：文本的特征抽取
    # text_count_demo()
    #第四五六部分，中文文本特征抽取（自动分词）
    # print(cut_word("人生苦短，我学python"))
    text_chinese_count_demo()
    #jieba断词测试
    # data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
    #         "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
    #         "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # tmp_list = []
    # for i in data:
    #     print(cut_word(i))
    #     tmp_list.append(cut_word(i))
    # print(tmp_list)
