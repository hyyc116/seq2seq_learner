#coding:utf-8
'''
对MAG的数据进行数据预处理

MAG的数据包括作者名消歧，本程序从所有的数据里面抽取某一个领域的论文数据，创建训练数据。



SELECT * FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog'
AND schemaname != 'information_schema';

 mag_core   | affiliations                         |
 mag_core   | journals                             |
 mag_core   | conference_series                    |
 mag_core   | conference_instances                 |
 mag_core   | papers                               |
 mag_core   | paper_resources                      |
 mag_core   | fields_of_study                      |
 mag_core   | related_field_of_study               |
 mag_core   | paper_urls                           |
 mag_core   | paper_abstract_inverted_index        |
 mag_core   | paper_author_affiliations            |
 mag_core   | authors                              |
 mag_core   | paper_citation_contexts              |
 mag_core   | paper_fields_of_study                |
 mag_core   | paper_languages                      |
 mag_core   | paper_recommendations                |
 mag_core   | paper_references                     |
 mag_core   | fields_of_study_children             |

'''
from basic_config import *

def read_data(field,tag):

    query_op = dbop()
    ## 首先从mag_core.fields_of_study找出和计算机相关的专业的id
    progress = 0
    field_id_name_dict = {}
    for field_of_study_id,normalized_name in query_op.query_database('select field_of_study_id,normalized_name from mag_core.fields_of_study'):

        progress+=1

        if progress%10000000==0:

            print('read field progress {} ..'.format(progress))


        if field.lower() in normalized_name.strip():
            field_id_name_dict[field_of_study_id] = normalized_name

    print('{} fields related to {}.'.format(len(field_id_name_dict),field))

    open('data/mag_{}_field_id_name_dict.json'.format(tag),'w').write(json.dumps(field_id_name_dict))

    print('paper fields mapping saved to data/mag_{}_field_id_name_dict.json'.format(tag))
    ## 然后，从mag_core.paper_fields_of_study找出对应id的论文

    paper_fields = defaultdict(list)

    sql = 'select paper_id,field_of_study_id,score from mag_core.paper_fields_of_study'

    progress=0

    for paper_id,field_of_study_id,score in query_op.query_database(sql):

        progress+=1

        if progress%10000000==0:
            print('read field paper progress {} ...'.format(progress))

        if field_id_name_dict.get(field_of_study_id,None) is None or score < 0.1:
            continue

        paper_fields[paper_id].append(field_of_study_id)

    print('{} papers in this fields ..'.format(len(paper_fields)))

    open('data/mag_{}_paper_fields.json'.format(tag),'w').write(json.dumps(paper_fields))
    print("paper fields mapping saved to data/mag_{}_paper_fields.json".format(tag))

    ## 根据得到的id列表，从mag_core.paper_author_affiliations 存储的是每一篇论文对应的作者的id以及作者顺序
    progress = 0

    author_papers = defaultdict(list)
    paper_authors = defaultdict(list)
    sql = 'select paper_id,author_id,author_sequence_number from mag_core.paper_author_affiliations'
    for paper_id,author_id,author_sequence_number in query_op.query_database(sql):

        progress+=1

        if progress%10000000==0:
            print('read author id {} ...'.format(progress))

        if paper_fields.get(paper_id,None) is None:
            continue

        author_papers[author_id].append([paper_id,author_sequence_number])
        paper_authors[paper_id].append([author_id,author_sequence_number])

    print('There are {} authors in this field..'.format(len(author_papers)))
    open('data/mag_{}_author_papers.json'.format(tag),'w').write(json.dumps(author_papers))
    print('author papers json saved to data/mag_{}_author_papers.json'.format(tag))
    open('data/mag_{}_paper_authors.json'.format(tag),'w').write(json.dumps(paper_authors))
    print('author papers json saved to data/mag_{}_paper_authors.json'.format(tag))
    print('Done')


# 根据作者的id，获取作者所有的论文
def read_paper_year(field,tag):
    paper_fields = json.loads(open('data/mag_{}_paper_fields.json'.format(tag)).read())
    query_op = dbop()
    sql = 'select paper_id,year from mag_core.papers'
    paper_year = {}
    progress = 0
    year_dis = defaultdict(int)
    for paper_id,year in query_op.query_database(sql):

        progress+=1

        if progress%10000000==0:
            print('Read paper year， progress {}, {} paper has year ...'.format(progress,len(paper_year)))

        if paper_fields.get(paper_id,None) is None:
            continue

        paper_year[paper_id] = year

        year_dis[int(year)]+=1

    print('Done, {}/{} paper has year ...'.format(len(paper_year),len(paper_fields)))
    open('data/mag_{}_paper_year.json'.format(tag),'w').write(json.dumps(paper_year))
    print('Data saved to data/mag_{}_paper_year.json'.format(tag))

    xs = []
    ys = []

    for x in sorted(year_dis.keys()):
        xs.append(x)
        ys.append(year_dis[x])

    plt.figure(figsize=(4,3))

    plt.plot(xs,ys)

    plt.xlabel("year")
    plt.ylabel("number of papers")

    plt.yscale('log')


    plt.tight_layout()

    plt.savefig('fig/mag_{}_paper_year_num_dis.png'.format(tag),dpi=400)

    print('Fig saved to fig/mag_{}_paper_year_num_dis.png'.format(tag))

## 作者以2012为界限，分为2012年以前以及2012年之后两部分，作者在2012年以前发表过论文就是2012年之前的作者，并且在2012年之后也发表过论文就是我们需要的作者
def filter_authors_by_year(tag,f_year,e_year):

    author_papers = json.loads(open('data/mag_{}_author_papers.json'.format(tag)).read())
    paper_year = json.loads(open('data/mag_{}_paper_year.json'.format(tag)).read())

    ## 根据上述规则对用户进行筛选
    reserved_authors = []
    reserved_paper_ids = []
    for author in author_papers:

        years = []
        papers = []
        for paper,sn in author_papers[author]:

            years.append(int(paper_year[paper]))
            papers.append(paper)

        if np.min(years) <f_year and np.max(years)>=e_year:

            reserved_authors.append(author)

            reserved_paper_ids.extend(papers)

    reserved_paper_ids = list(set(reserved_paper_ids))

    print('{} Authors reserved, with {}.'.format(len(reserved_authors),len(reserved_paper_ids)))

    open('data/mag_{}_reserved_authors.txt'.format(tag),'w').write('\n'.join(reserved_authors))

    print('Data saved to data/mag_{}_reserved_authors.txt'.format(tag))


    open('data/mag_{}_reserved_papers.txt'.format(tag),'w').write('\n'.join(reserved_paper_ids))
    print('Reserved paper ids saved to data/mag_{}_reserved_papers.txt'.format(tag))

## 将2012年的论文 预测其2013年后 2014年后 2017年被2012年之前的作者引用的次数
def paper_author_cits(tag):

    ## 改领域所有论文的发表时间
    paper_year = json.loads(open('data/mag_{}_paper_year.json'.format(tag)).read())

    paper_authors = json.loads(open('data/mag_{}_paper_authors.json'.format(tag)).read())

    ## 根据是否存在发表时间来判断，领域内引用
    query_op = dbop()

    sql = 'select paper_id,paper_reference_id from mag_core.paper_references'
    progress = 0
    paper_refs = []
    for paper_id,paper_reference_id in query_op.query_database(sql):

        progress +=1

        if progress%10000000==0:
            print('progress {}, {} citations ...'.format(progress,len(paper_refs)))

        if paper_year.get(paper_id,None) is None:
            continue

        if paper_year.get(paper_reference_id,None) is None:
            continue

        if paper_authors.get(paper_id,None) is None:
            continue

        if paper_authors.get(paper_reference_id,None) is None:
            continue

        paper_refs.append('{},{}'.format(paper_id,paper_reference_id))

    open('data/mag_{}_paper_cits.txt'.format(tag),'w').write('\n'.join(paper_refs))
    print('{} citation relations saved to data/mag_{}_paper_cits.txt'.format(len(paper_refs),tag))



## 根据N年的历史进行预测N年的结果
def filter_papers(tag):

    ## paper year
    paper_year = json.loads(open('data/mag_{}_paper_year.json'.format(tag)).read())

    ##加载作者文章
    author_papers = json.loads(open('data/mag_{}_author_papers.json'.format(tag)).read())
    ## 文章与作者的对应关系
    paper_authors = json.loads(open('data/mag_{}_paper_authors.json'.format(tag)).read())
    print('Length of paper authors {}.'.format(len(paper_authors)))

    ## 根据文章的被引用情况对2012年的论文进行筛选
    _2012_papers = set([paper_id.strip() for paper_id in paper_year.keys() if int(paper_year[paper_id])==2012])
    reserved_paper_ids = set([paper_id.strip() for paper_id in open('data/mag_{}_reserved_papers.txt'.format(tag))])

    ## 统计2012年论文的引用总次数
    _2012_paper_cn = defaultdict(int)
    ## 统计2012年论文每年的引用论文
    _2012_paper_year_cn = defaultdict(lambda:defaultdict(list))

    ## 统计2012年的论文被各个作者引用的情况
    _2012_paper_year_author = defaultdict(lambda:defaultdict(list))

    ## 加载引用关系
    for line in open('data/mag_{}_paper_cits.txt'.format(tag)):

        line = line.strip()

        pid,cited_pid = line.split(',')

        if cited_pid in _2012_papers:

            if paper_year.get(pid,0)==0:
                continue

            _2012_paper_cn[cited_pid]+=1

            _2012_paper_year_cn[cited_pid][int(paper_year[str(pid)])].append(pid)

            for author,ix in paper_authors[pid]:

                _2012_paper_year_author[cited_pid][int(paper_year[str(pid)])].append(author)


    ## 1. 首先根据被引用的总次数进行过滤,总次数要求大于5
    _2012_papers_used = [paper_id for paper_id in _2012_papers if _2012_paper_cn.get(paper_id,0)>5]
    print('Number of paper after filtering is {}.'.format(len(_2012_papers_used)))

    ## 统计分析随着时间 2012年论文被引用次数在发表后不同的时间中，被作者引用以及全部引用的关系

    ## 根据引用关系，生成每一篇论文在各个年份被作者引用的总次数
    y_percents =defaultdict(list)
    lines = []

    total = len(_2012_papers_used)
    progress = 0
    for paper_id in _2012_papers_used:

        progress+=1

        if progress%10000==0:
            print('Gen data progress {}/{}, number of samples {} ...'.format(progress,total,len(lines)))

        cn = _2012_paper_cn[paper_id]
        author_ts  = defaultdict(int)
        t_cn_t = 0
        for  y in [2012,2013,2014,2015,2016,2017]:

            t_cn = _2012_paper_year_cn[paper_id].get(y,[])

            c_b_a = 0
            for p in t_cn:
                if p in reserved_paper_ids:
                    c_b_a+=1

                    ## 获得这篇论文的作者
                    c_aus = paper_authors[p]

                    ## 这位作者到今年引用的次数
                    for ca,ix in c_aus:

                        if ix==1:
                            author_ts[ca]+=1

            ## 每一年输出这个作者引用的次数，
            for cau in author_ts.keys():
                c_au_cn = author_ts[cau]
                line = '{},{},{},{}'.format(paper_id,y,cau,c_au_cn)
                lines.append(line)

            t_cn_t+=len(t_cn)
            ## 今年被引用的总次数，因为合作的存在导致被作者引用次数的和大于总次数
            lines.append('{},{},{},{}'.format(paper_id,y,'ALL',t_cn_t))
            # lines.append(line)
            

            ## 查看被已有作者引用的次数与总次数之间的关系
            if cn>20 & len(t_cn)>0:
                p = c_b_a/float(len(t_cn))
                y_percents[y].append(p)

    open('data/mag_{}_raw_data_paper_author.txt'.format(tag),'w').write('\n'.join(lines))
    print('{} samples saved to data/mag_{}_raw_data_paper_author.txt'.format(len(lines),tag))

    xs = []
    ys = []
    for y in sorted(y_percents.keys()):

        percents = y_percents[y]
        xs.append(y)
        ys.append(percents)
        # ys2.append(np.median(limits))

    # print('Total number of papers:{} ...'.format(num_count))
    plt.figure(figsize=(4,3))
    plt.boxplot(ys)
    plt.xlabel("year")
    plt.ylabel('percents')
    plt.xticks([x+1 for x in range(len(xs))],xs)
    plt.tight_layout()
    plt.savefig('fig/_2012_paper_cit_limit.png',dpi=400)
    print('Number of paper citations and limitations.')


##  生成训练数据
def gen_data(tag):

    ## 2012年的id，在2012年，2013年，2016年被引用的总次数以及被某位作者引用的次数

    ## id, year, author, number

    ## 12121, 2012, ALL, 10 // ALL表示所有人，包括新作者
    ## 12121, 2012, a1, 2 // 这篇文章被作者a1在2012年引用了两次
    print('Load 2012 paper data ...')
    _2012_papers = set([paper_id.strip() for paper_id in open('data/mag_{}_2012_papers.txt'.format(tag))])

    print('Load author paper ...')
    reserved_paper_ids = set([paper_id.strip() for paper_id in open('data/mag_{}_reserved_papers.txt'.format(tag))])

    ##加载作者文章
    print('Load author paper data ...')

    author_papers = json.loads(open('data/mag_{}_author_papers.json'.format(tag)).read())

    paper_author = defaultdict(list)

    for author in author_papers.keys():

        for paper,ix in author_papers[author]:

            paper_author[paper].append([author,ix])

    ## paper year
    print('Load paper year data ...')
    paper_year = json.loads(open('data/mag_{}_paper_year.json'.format(tag)).read())

    print('Load citing relations ...')

    paper_year_citings =  defaultdict(lambda:defaultdict(list))
    for line in open('data/mag_{}_paper_cits.txt'.format(tag)):

        pid,cited_pid = line.strip().split(',')

        pyear = paper_year.get(pid,None)

        if pyear is None:
            continue

        if cited_pid in _2012_papers:

            paper_year_citings[cited_pid][int(pyear)].append(pid)

    print('Length of paper year citings {} ...'.format(len(paper_year_citings)))


    ##对于2012发表的论文
    print('gen data ...')
    lines = []
    progress=0
    for pid in paper_year_citings.keys():

        progress+=1

        if progress%10000==0:

            print('Number of training data {} ..'.format(len(lines)))

        ## 按照年份将作者被各个作者引用的次数进行记录
        author_times = defaultdict(int)
        total = 0
        for year in sorted(paper_year_citings[pid].keys()):
            citings = paper_year_citings[pid][year]

            ## 每一篇引证文献
            for citing_pid in citings:

                if citing_pid not in reserved_paper_ids:
                    continue

                authors = paper_author[citing_pid]

                for author,ix in authors:

                    author_times[author]+=1

            for author in author_times.keys():

                line = '{},{},{},{}'.format(pid,year,author,author_times[author])
                lines.append(line)

            total+= len(citings)
            line = '{},{},{},{}'.format(pid,year,'ALL',total)
            lines.append(line)

    open('data/pid_author_year_num.txt','w').write('\n'.join(lines))
    print('Data Saved to data/pid_author_year_num.txt.')





if __name__ == '__main__':
    field = 'computer science'
    tag = 'cs'
    # read_data(field,tag)
    # read_paper_year(field,tag)
    # filter_authors_by_year(tag,2012,2017)
    # paper_author_cits(tag)

    filter_papers(tag)

    # gen_data(tag)

