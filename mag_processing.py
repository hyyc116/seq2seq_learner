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

        if progress%100000==0:

            print('read field progress {} ..'.format(progress))


        if field.lower() in normalized_name:
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

        if progress%100000==0:
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
    sql = 'select paper_id,author_id,author_sequence_number from mag_core.paper_author_affiliations'
    for paper_id,author_id,author_sequence_number in query_op.query_database(sql):

        progress+=1

        if progress%100000==0:
            print('read author id {} ...'.format(progress))

        if paper_fields.get(paper_id,None) is None:
            continue

        author_papers[author_id].append([paper_id,author_sequence_number])

    print('There are {} authors in this field..'.format(len(author_papers)))
    open('data/mag_{}_author_papers.json'.format(tag),'w').write(json.dumps(author_papers))
    print('author papers json saved to data/mag_{}_author_papers.json'.format(tag))

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

        if progress%1000000==0:
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
def filter_authors_by_year(tag,f_year):

    author_papers = json.loads(open('data/mag_{}_author_papers.json'.format(tag)).read())

    paper_year = json.loads(open('data/mag_{}_paper_year.json'.format(tag)).read())

    ## 根据上述规则对用户进行筛选
    reserved_authors = []
    for author in author_papers:

        years = []
        for paper,sn in author_papers[author]:

            years.append(int(paper_year[paper]))

        if np.min(years) <f_year and np.max(years)>=2017:

            reserved_authors.append(author)

    print('{} Authors reserved.'.format(len(reserved_authors)))

    open('data/mag_{}_reserved_authors.txt'.format(tag),'w').write('\n'.join(reserved_authors))

    print('Data saved to data/mag_{}_reserved_authors.txt'.format(tag))

## 将2012年的论文 预测其2013年后 2014年后 2017年被2012年之前的作者引用的次数
def paper_author_cits(tag):

    paper_year = json.loads(open('data/mag_{}_paper_year.json'.format(tag)).read())

    paper_ids= []
    for paper in paper_year.keys():

        year  = int(paper_year[paper])

        if year==2012:
            paper_ids.append(paper)

    author_papers = json.loads(open('data/mag_{}_author_papers.json'.format(tag)).read())

    open('data/mag_{}_2012_papers.txt','w').write('\n'.join(paper_ids))

    reserved_authors = [ author.strip() for author in  open('data/mag_{}_reserved_authors.txt'.format(tag))]

    reserved_paper_ids = []
    for author in reserved_authors:

        reserved_paper_ids.extend([p for p,_ in author_papers[author]])

    paper_ids = set(paper_ids)

    print('Number of papers published in 2012 is {}.'.format(len(paper_ids)))

    reserved_paper_ids = set(reserved_paper_ids)

    open('data/mag_{}_reserved_papers.txt','w').write('\n'.join(reserved_paper_ids))


    print('Number of papers of auhtors is {}.'.format(len(reserved_paper_ids)))

    ## 根据paper_ids以及已存在的
    query_op = dbop()

    sql = 'select paper_id,paper_reference_id from mag_core.paper_references'
    progress = 0
    paper_refs = []
    for paper_id,paper_reference_id in query_op.query_database(sql):

        progress +=1

        if progress%10000000==0:
            print('progress {}, {} citations ...'.format(progress,len(paper_refs)))

        if paper_id in reserved_paper_ids or  paper_reference_id in paper_ids:

            paper_refs.append('{},{}'.format(paper_id,paper_reference_id))

    open('data/mag_{}_paper_cits.png'.format(tag),'w').write('\n'.join(paper_refs))
    print('{} citation relations saved to data/mag_{}_paper_cits.png'.format(len(paper_refs),tag))


if __name__ == '__main__':
    field = 'computer science'
    tag = 'cs'
    # read_data(field,tag)
    # read_paper_year(field,tag)
    # filter_authors_by_year(tag,2012)
    paper_author_cits(tag)


