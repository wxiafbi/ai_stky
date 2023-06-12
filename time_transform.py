import time

'''
时间戳转换成日期格式
1）将时间时间戳转换成时间元组
2）通过strftime(）将时间元组转换成格式化时间
'''
def timestamp_to_format(timestamp=None, format='%Y-%m-%d %H:%M:%S'):
    # try:
    if timestamp:
        time_tuple = time.localtime(timestamp)
        #print('time_tuple:', time_tuple)
        # print('type(time_tuple):',type(time_tuple))
        res = time.strftime(format, time_tuple)
    else:
        res = time.strftime(format)
    return res
'''
将格式化时间转换为时间戳
1）将日期转换成时间元组
2）转换成时间戳
'''
def timeformat_to_timestamp(timeformat=None,format = '%Y-%m-%d %H:%M:%S'):
    # try:
    if timeformat:
        time_tuple = time.strptime(timeformat,format)
        res = time.mktime(time_tuple) #转成时间戳
    else:
        res = time.time()        #获取当前时间戳
    return int(res)
    #print("timeformat_to_timestamp('2018-04-12 09:09:45')的时间戳：",timeformat_to_timestamp('2018-04-12 09:09:45'))

if __name__ == '__main__':
    ac = timestamp_to_format(1683921658)

    # print(ac,'\t',at)
    print(ac)
    format = input("请输入时间：")
    at = timeformat_to_timestamp(format)
    print(at)
    # print('时间戳1234567转换成日期格式为：',timestamp_to_format(1234567))
