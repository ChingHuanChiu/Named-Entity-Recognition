

def get_inference(extractor, content):
    
    result_dict = extractor.extract(content)
    del result_dict['decode_sequence']
    
    return {'content': result_dict['content'],
            'Thing': result_dict['Thing'],
            'person': result_dict['Person'],  
            'Location': result_dict['Location'],
            'Time': result_dict['Time'],
            'Metric': result_dict['Metric'],
            'Organization': result_dict['Organization'],
            'Abstract': result_dict['Abstract'],
            'Physical': result_dict['Physical'],
            'Term': result_dict['Term'],         
           }
