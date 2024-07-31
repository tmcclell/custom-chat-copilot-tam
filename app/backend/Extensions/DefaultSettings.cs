﻿namespace MinimalApi.Extensions;

public static class DefaultSettings
{
    public static int MaxRequestTokens = 6000;
    public static int KNearestNeighborsCount = 3;

    public static PromptExecutionSettings AISearchRequestSettings = new()
    {
        ExtensionData = new Dictionary<string, object>()
        {
            { "max_tokens", 1500 },
            { "temperature", 0.0 },
            { "top_p", 1 }
        }
    };
    public static PromptExecutionSettings AIChatRequestSettings = new()
    { 
        ExtensionData = new Dictionary<string, object>()
        {
            { "max_tokens", 4096 },
            { "temperature", 0.0 },
            { "top_p", 1 },
        }
    };

    public static string DocumentRetrievalPluginName = "DocumentRetrieval";

    public static string CosmosDBDatabaseName = "ChatHistory";
    public static string CosmosDBCollectionName = "ChatTurn";
    public static string CosmosDBUserDocumentsCollectionName = "UserDocuments";

    public static string GenerateSearchQueryPluginName = "GenerateSearchQuery";
    public static string GenerateSearchQueryPluginQueryFunctionName = "GenerateSearchQuery";

    public static string ChatPluginName = "Chat";
    public static string ChatPluginFunctionName = "Chat";
}
