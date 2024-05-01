from LER_model import LER_model


ler_model = LER_model()

text = "The electrical conductivity decrease from 29.3%IACS to 14.5%IACS (the electrical conductivity of annealed pure copper is equal to 100%IACS) as the TiB2 content increase from 0wt% to 10wt%."
keywords = 'A'
print(type(keywords))
print(ler_model.LER_regular(text,keywords))
keywords = ['Cu','conductivity','IACS']
print(type(keywords))
print(ler_model.LER_regular(text,keywords))

text = "The electrical conductivity decrease from 29.3%IACS to 14.5%IACS (the electrical conductivity of annealed pure copper is equal to 100%IACS) as the TiB2 content increase from 0wt% to 10wt%."

keyword_list = ['Cu,copper,wt%','conductivity,tensile','IACS,MPa']


if len(keyword_list[1].split(',')) > 1:
    # print(keywords)
    if len(keyword_list[1].split(',')) == len(keyword_list [2].split(',')):
        relation_list = keyword_list[1].split(",")
        ent_2_list = keyword_list[2].split(",")
        ent_1_list = keyword_list[0].split(",")
        for i in range(len(relation_list)):
            for j in range(len(ent_1_list)):
                keywords = [ent_1_list[j],relation_list[i],ent_2_list[i]]
                print(ler_model.LER_regular(text,keywords))


