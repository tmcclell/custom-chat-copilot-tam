﻿// Copyright (c) Microsoft. All rights reserved.

using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Azure.AI.OpenAI;
using Azure.Search.Documents;
using MinimalApi.Extensions;
using MinimalApi.Services.Profile;
using TiktokenSharp;

namespace MinimalApi.Services.Search;

public class SearchLogic<T> where T : IKnowledgeSource
{
    private readonly SearchClient _searchClient;
    private readonly OpenAIClient _openAIClient;
    private readonly string _embeddingModelName;
    private readonly string _embeddingFieldName;
    private readonly List<string> _selectFields;
    private readonly int _documentFilesCount;

    public SearchLogic(OpenAIClient openAIClient, SearchClientFactory factory, string indexName, string embeddingModelName, string embeddingFieldName, List<string> selectFields, int documentFilesCount)
    {
        _searchClient = factory.GetOrCreateClient(indexName);
        _openAIClient = openAIClient;
        _embeddingModelName = embeddingModelName;
        _embeddingFieldName = embeddingFieldName;
        _selectFields = selectFields;
        _documentFilesCount = documentFilesCount;
    }

    public async Task<KnowledgeSourceSummary> SearchAsync(string query, KernelArguments arguments)
    {
        // Generate the embedding for the query  
        var queryEmbeddings = await GenerateEmbeddingsAsync(query, _openAIClient);


        // Configure the search options
        var searchOptions = new SearchOptions
        {
            Size = _documentFilesCount,
            VectorSearch = new()
            {
                Queries = { new VectorizedQuery(queryEmbeddings.ToArray()) { KNearestNeighborsCount = DefaultSettings.KNearestNeighborsCount, Fields = { _embeddingFieldName } } }
            }
        };

        foreach (var field in _selectFields)
        {
            searchOptions.Select.Add(field);
        }
     
        if (arguments.ContainsName(ContextVariableOptions.SelectedDocument))
        {
            var userId = arguments[ContextVariableOptions.UserId] as string;
            var sessionId = arguments[ContextVariableOptions.SessionId] as string;
            var sourcefile = arguments[ContextVariableOptions.SelectedDocument] as string;
            searchOptions.Filter = $"entra_id eq '{userId}' and session_id eq '{sessionId}' and sourcefile eq '{sourcefile}'";
        }

        // Perform the search and build the results
        try
        {
            var response = await _searchClient.SearchAsync<T>(query, searchOptions);
            var list = new List<T>();
            // foreach (var result in response.Value.GetResults())
            // {
            //     list.Add(result.Document);
            // }

            // Filter the results by the maximum request token size
            var sourceSummary = FilterByMaxRequestTokenSize(list, DefaultSettings.MaxRequestTokens);
            return sourceSummary;
        }
        catch (Exception ex)
        {
            // Handle the exception gracefully
            Debug.WriteLine($"An error occurred during the search: {ex.Message}");
            return null; // Or handle the error in an appropriate way for your application
        }
        // var list = new List<T>();
        // foreach (var result in response.Value.GetResults())
        // {
        //     list.Add(result.Document);
        // }

        // /// Filter the results by the maximum request token size
        // var sourceSummary = FilterByMaxRequestTokenSize(list, DefaultSettings.MaxRequestTokens);
        // return sourceSummary;
    }

    private KnowledgeSourceSummary FilterByMaxRequestTokenSize(IReadOnlyList<T> sources, int maxRequestTokens)
    {
        int sourceSize = 0;
        int tokenSize = 0;
        var documents = new List<IKnowledgeSource>();
        var tikToken = TikToken.EncodingForModel("gpt-3.5-turbo");
        var sb = new StringBuilder();
        foreach (var document in sources)
        {
            var text = document.FormatAsOpenAISourceText();
            sourceSize += text.Length;
            tokenSize += tikToken.Encode(text).Count;
            if (tokenSize > maxRequestTokens)
            {
                break;
            }
            documents.Add(document);
            sb.AppendLine(text);
        }
        return new KnowledgeSourceSummary(sb.ToString(), documents);
    }

    private async Task<ReadOnlyMemory<float>> GenerateEmbeddingsAsync(string text, OpenAIClient openAIClient)
    {
        var response = await openAIClient.GetEmbeddingsAsync(new EmbeddingsOptions(_embeddingModelName, new[] { text }));
        return response.Value.Data[0].Embedding;
    }
}
