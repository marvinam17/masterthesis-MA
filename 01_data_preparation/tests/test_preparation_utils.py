from preparation_utils import convert_beir_dbpedia_entity_v2_to_df, add_wikipedia_page_id, add_wikidata_id, preprocess_wikidataLinks, preprocess_wikiPageLinks
import pandas as pd
import pandas as pd
from unittest.mock import patch, mock_open


def test_convert_beir_dbpedia_entity_v2_to_df():
    # Test input
    test_corpus = {
        "<dbpedia:Test1>": {"text": "test1 text", "title": "Test 1"},
        "<dbpedia:Test2>": {"text": "test2 text", "title": "Test 2"},
        "not_dbpedia": {"text": "should be filtered", "title": "Filter"},
        "<dbpedia:List_of_fictional_extraterrestrials_by_form>": {"text": "should be filtered", "title": "Filter"}
    }

    # Run function
    df = convert_beir_dbpedia_entity_v2_to_df(test_corpus)

    # Verify results
    assert len(df) == 2
    assert list(df.columns) == ['id', 'text', 'title']
    assert df['id'].tolist() == ['<dbpedia:Test1>', '<dbpedia:Test2>']
    assert df['text'].tolist() == ['test1 text', 'test2 text']
    assert df['title'].tolist() == ['Test 1', 'Test 2']
    assert all(df['id'].str.contains('dbpedia:'))
    assert '<dbpedia:List_of_fictional_extraterrestrials_by_form>' not in df['id'].values

def test_add_wikipedia_page_id():
    # Create test input dataframes
    test_df = pd.DataFrame({
        'id': ['<dbpedia:Test1>', '<dbpedia:Beta_(programming_language)>'],
        'text': ['text1', 'text3']
    })
    
    test_mapping_df = pd.DataFrame({
        'id': ['<dbpedia:Test1>'],
        'page_id': [12345]
    })

    # Run function
    result_df = add_wikipedia_page_id(test_df, test_mapping_df)
    
    # Assertions
    assert isinstance(result_df, pd.DataFrame)
    assert 'page_id' in result_df.columns
    assert result_df.loc[result_df['id'] == '<dbpedia:Test1>', 'page_id'].iloc[0] == 12345
    assert result_df.loc[result_df['id'] == '<dbpedia:Beta_(programming_language)>', 'page_id'].iloc[0] == 135868
    assert len(result_df) == len(test_df)
    assert not result_df['page_id'].isna().any()


@patch('preparation_utils.get_wikidata_id_from_dbpedia_dump')
@patch('preparation_utils.execute_sparql_query')
@patch('preparation_utils.postprocess_results')
def test_add_wikidata_id(mock_postprocess, mock_execute_sparql, mock_get_mapping):
    # Sample DataFrame
    sample_df = pd.DataFrame({
        'id': ['Q1', 'Q2', 'Q3'],
        'page_id': [123, 456, 789]
    })

    # Mocked data
    mock_mapping_dbpedia_wikidata_ids = {'Q1': 'Q123', 'Q2': 'Q456'}
    mock_sparql_results = [{'wikiPageID': '789', 'wikidata_concept': 'Q789'}]
    mock_postprocessed_results = {789: 'Q789'}

    # Set up the mocks
    mock_get_mapping.return_value = mock_mapping_dbpedia_wikidata_ids
    mock_execute_sparql.return_value = mock_sparql_results
    mock_postprocess.return_value = mock_postprocessed_results

    # Call the function
    result_df = add_wikidata_id(sample_df, save_sparql_mapping=False)

    # Assertions
    assert 'item_id' in result_df.columns
    assert result_df['item_id'].tolist() == ['Q123', 'Q456', 'Q789']
    mock_get_mapping.assert_called_once_with("../03_data/raw_data/wikidata/sameas_all_wikis_wikidata.ttl")
    mock_postprocess.assert_called_once_with(mock_sparql_results)

    # Check if the 'item_id_sparql' column is dropped
    assert 'item_id_sparql' not in result_df.columns


def test_preprocess_wikiPageLinks(tmp_path):
    # Create test data
    test_df = pd.DataFrame({
        'id': ['<dbpedia:Test1>', '<dbpedia:Test2>', '<dbpedia:Test3>']
    })
    
    # Create test input file
    test_file = tmp_path / "test_links.ttl"
    test_content = """<http://dbpedia.org/resource/Test1> <http://www.w3.org/2000/01/rdf-schema#seeAlso> <http://dbpedia.org/resource/Test2> .
<http://dbpedia.org/resource/Test2> <http://www.w3.org/2000/01/rdf-schema#seeAlso> <http://dbpedia.org/resource/Test1> .
<http://dbpedia.org/resource/Test1> <http://www.w3.org/2000/01/rdf-schema#seeAlso> <http://dbpedia.org/resource/Test3> .
<http://dbpedia.org/resource/Test1> <http://www.w3.org/2000/01/rdf-schema#seeAlso> <http://dbpedia.org/resource/Test1> .
<http://dbpedia.org/resource/Test4> <http://www.w3.org/2000/01/rdf-schema#seeAlso> <http://dbpedia.org/resource/Test5> ."""
    test_file.write_text(test_content)

    # Process links
    result = preprocess_wikiPageLinks(test_df, str(test_file))

    # Check results
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # Should only have unique relations - Test1-Test2 and Test1-Test3
    assert set(result.columns) == {'source', 'target'}
    assert result.iloc[0]['source'] != result.iloc[0]['target']  # No self-relations
    assert len(result[result['source'] == result['target']]) == 0  # No self-relations    


# Sample file content
sample_file_content = """# This is a sample file
<http://wikidata.dbpedia.org/resource/Q123> <http://www.w3.org/2002/07/owl#sameAs> <http://wikidata.dbpedia.org/resource/Q456> .
<http://wikidata.dbpedia.org/resource/Q456> <http://www.w3.org/2002/07/owl#sameAs> <http://wikidata.dbpedia.org/resource/Q123> .
<http://wikidata.dbpedia.org/resource/Q123> <http://www.w3.org/2002/07/owl#sameAs> <http://wikidata.dbpedia.org/resource/Q789> .
<http://wikidata.dbpedia.org/resource/Q789> <http://www.w3.org/2002/07/owl#sameAs> <http://wikidata.dbpedia.org/resource/Q123> .
<http://wikidata.dbpedia.org/resource/Q456> <http://www.w3.org/2002/07/owl#sameAs> <http://wikidata.dbpedia.org/resource/Q789> .
<http://wikidata.dbpedia.org/resource/Q789> <http://www.w3.org/2002/07/owl#sameAs> <http://wikidata.dbpedia.org/resource/Q456> .
"""

@patch('builtins.open', new_callable=mock_open, read_data=sample_file_content)
def test_preprocess_wikidataLinks(mock_file):
    # Sample DataFrame
    sample_df = pd.DataFrame({
        'item_id': [123, 456, 789]
    })
    # Call the function
    result_df = preprocess_wikidataLinks(sample_df, path="dummy_path")

    # Assertions
    assert 'source' in result_df.columns
    assert 'target' in result_df.columns
    assert len(result_df) == 3  # There should be 3 unique relations/
    assert result_df['source'].tolist() == [0,0,1]
    assert result_df['target'].tolist() == [1,2,2]

    # Check if the file was opened correctly
    mock_file.assert_called_once_with("dummy_path")