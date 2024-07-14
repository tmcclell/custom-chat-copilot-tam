using System.ComponentModel;
using System.Text.Json.Serialization;
using Azure;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Models;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;

namespace MinimalApi.Services.Search;
public class SearchPlugin
{
    private readonly ITextEmbeddingGenerationService _textEmbeddingGenerationService;
    private readonly SearchIndexClient _indexClient;
    private readonly OpenAIClient _openAIClient;
    private readonly string _embeddingModelName;
    private readonly string _embeddingFieldName;
    

    public SearchPlugin(ITextEmbeddingGenerationService textEmbeddingGenerationService, OpenAIClient openAIClient, SearchIndexClient indexClient, string embeddingModelName, string embeddingFieldName)
    {
        _textEmbeddingGenerationService = textEmbeddingGenerationService;
        _indexClient = indexClient;
        _openAIClient = openAIClient;
        _embeddingModelName = embeddingModelName;
        _embeddingFieldName = embeddingFieldName;
    }

    [KernelFunction("Search")]
    [Description("Search for a document similar to the given query.")]
    public async Task<string> SearchAsync(string query)
    {
        // Convert string query to vector
        ReadOnlyMemory<float> embedding = await _textEmbeddingGenerationService.GenerateEmbeddingAsync(query);

        // Get client for search operations
        SearchClient searchClient = _indexClient.GetSearchClient("default-collection");

        // Configure request parameters
        VectorizedQuery vectorQuery = new(embedding);
        vectorQuery.Fields.Add("vector");

        SearchOptions searchOptions = new() { VectorSearch = new() { Queries = { vectorQuery } } };

        // Perform search request
        Response<SearchResults<IndexSchema>> response = await searchClient.SearchAsync<IndexSchema>(searchOptions);

        // Collect search results
        await foreach (SearchResult<IndexSchema> result in response.Value.GetResultsAsync())
        {
            return result.Document.Chunk; // Return text from first result
        }

        return string.Empty;
    }

    private sealed class IndexSchema
    {
        [JsonPropertyName("chunk")]
        public string Chunk { get; set; }

        [JsonPropertyName("text_vector")]
        public ReadOnlyMemory<float> Vector { get; set; }
    }

    public interface ITextEmbeddingGenerationService
    {
        Task<ReadOnlyMemory<float>> GenerateEmbeddingAsync(string text);
    }
    
}