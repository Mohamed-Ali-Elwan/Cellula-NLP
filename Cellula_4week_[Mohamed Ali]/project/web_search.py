from langchain_tavily import TavilySearch
api_key="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"
class WebSearch:
    
    @staticmethod
    def search(query: str, num_results: int = 10):
       
        tavily_search = TavilySearch(api_key=api_key, search_depth= "advanced")
        results = tavily_search.search(query, num_results)
        
        return results