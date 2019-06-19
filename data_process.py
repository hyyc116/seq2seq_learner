#coding:utf-8
'''
将小说的每一个章节的内容以及标题进行抽取

'''
import sys
# from imp import reload
# reload(sys)

# sys.setdefaultencoding('utf-8')

## 每一本小数的地址
def extract_title_content(path):

    all_lines = []

    last_title = None
    last_index= -1

    t_index = -1
    title = None
    content = ''

    isT = False
    for line in open(path,encoding='utf-8'):

        line = line.strip()

        if '第' in line and len(line)<30:

            # print(line)

            d_index = line.index('第')

            if "章" in line:
                ## 简单的找标题

                # print(line)
                isT = True

                z_index  = line.index('章')

                c_order = line[d_index+1:z_index]

                if len(c_order)>5:
                    c_order = c_order.split('第')[-1]

                title = line[z_index+1:].strip()

                if '（' in title:

                    title = title[:title.index('（')]

                if '(' in title:

                    title = title[:title.index('(')]


                t_index = c_order

            if '节' in line: 

                isT = True

                z_index = line.index('节')

                c_order = line[d_index+1:z_index]
                if len(c_order)>5:
                    c_order = c_order.split('第')[-1]

                title = line[z_index+1:].strip()
                if '（' in title:

                    title = title[:title.index('（')]

                if '(' in title:

                    title = title[:title.index('(')]

                t_index = c_order

        else:

            isT=False

            content+=line


        ## 如果是标题，那么保存上一章节的内容以及标题
        if isT:
            last_title = title
            last_index = t_index
            if last_title is not None and content!='':

                data_line = t_index+"==========================="+last_title+'==========================='+content
                # print(data_line)
                # data_line = data_line.encode('utf-8')
                all_lines.append(data_line)
                content = ''


    f = open('data/train/train.txt','a',encoding='utf-8')

    f.write(str('\n'.join(all_lines)+'\n'))

    print('%d chapters saved.' % len(all_lines))
            



if __name__ == '__main__':
    extract_title_content('data/novel/1.txt')
    extract_title_content('data/novel/2.txt')
    extract_title_content('data/novel/3.txt')
    extract_title_content('data/novel/4.txt')
    extract_title_content('data/novel/5.txt')
    extract_title_content('data/novel/6.txt')
    extract_title_content('data/novel/7.txt')
    extract_title_content('data/novel/8.txt')

    