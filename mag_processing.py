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

    open('data/mag_{}_field_id_name_dict.json'.format(tag),'w').write(json.dumps(mag_field_id_name_dict))
    
    print('paper fields mapping saved to data/mag_{}_field_id_name_dict.json'.format(tag))
    ## 然后，从mag_core.paper_fields_of_study找出对应id的论文

    paper_fields = defaultdict(list)

    sql = 'select paper_id,field_of_study_id,score from mag_core.paper_fields_of_study'

    progress=0

    for paper_id,field_of_study_id,score in query_op.query_database(sql):

        progress+=1

        if progress%100000==0:
            print('read field paper progress {} ...'.format(progress))

        if field_id_name_dict.get(field_of_study_id,None) is None:
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

        if progress&100000==0:
            print('read author id {} ...'.format(progress))

        if paper_fields.get(paper_id,None) is None:
            continue

        author_papers[author_id].append([paper_id,author_sequence_number])

    print('There are {} authors in this field..'.format(len(author_papers)))
    open('data/mag_{}_author_papers.json'.format(tag),'w').write(author_papers)
    print('author papers json saved to data/mag_{}_author_papers.json'.format(tag))

    print('Done')

    # 根据作者的id，获取作者所有的论文





   

if __name__ == '__main__':
    read_data('computer science','cs')