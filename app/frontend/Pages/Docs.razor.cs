﻿// Copyright (c) Microsoft. All rights reserved.

namespace ClientApp.Pages;

public sealed partial class Docs : IDisposable
{
    private const long MaxIndividualFileSize = 1_024 * 1_024 * 10;

    private MudForm _form = null!;
    private MudFileUpload<IReadOnlyList<IBrowserFile>> _fileUpload = null!;
    private Task _getDocumentsTask = null!;
    private bool _isLoadingDocuments = false;
    private string _filter = "";

    // Store a cancelation token that will be used to cancel if the user disposes of this component.
    private readonly CancellationTokenSource _cancellationTokenSource = new();
    private readonly HashSet<DocumentSummary> _documents = [];

    [Inject]
    public required ApiClient Client { get; set; }

    [Inject]
    public required ISnackbar Snackbar { get; set; }

    [Inject]
    public required ILogger<Docs> Logger { get; set; }

    [Inject]
    public required IJSRuntime JSRuntime { get; set; }

    private bool FilesSelected => _fileUpload is { Files.Count: > 0 };

    protected override void OnInitialized()
    {
        // Instead of awaiting this async enumerable here, let's capture it in a task
        // and start it in the background. This way, we can await it in the UI.
        _getDocumentsTask = GetDocumentsAsync();
    }

    private bool OnFilter(DocumentSummary document) => document is not null
        && (string.IsNullOrWhiteSpace(_filter) || document.Name.Contains(_filter, StringComparison.OrdinalIgnoreCase));

    private async Task GetDocumentsAsync()
    {
        _isLoadingDocuments = true;

        try
        {
            var documents = await Client.GetDocumentsAsync(_cancellationTokenSource.Token).ToListAsync();
            _documents.Clear();
            foreach (var document in documents)
            {
                _documents.Add(document);
            }
        }
        finally
        {
            _isLoadingDocuments = false;
            StateHasChanged();
        }
    }
    private async Task RefreshAsync()
    {
        await GetDocumentsAsync();
    }
    private async Task SubmitFilesForUploadAsync()
    {
        if (_fileUpload is { Files.Count: > 0 })
        {
            var cookie = await JSRuntime.InvokeAsync<string>("getCookie", "XSRF-TOKEN");
            var result = await Client.UploadDocumentsAsync(_fileUpload.Files, MaxIndividualFileSize, cookie);

            Logger.LogInformation("Result: {x}", result);

            if (result.IsSuccessful)
            {
                Snackbar.Add(
                    $"Uploaded {result.UploadedFiles.Length} documents.",
                    Severity.Success,
                    static options =>
                    {
                        options.ShowCloseIcon = true;
                        options.VisibleStateDuration = 10_000;
                    });

                await _fileUpload.ResetAsync();
            }
            else
            {
                Snackbar.Add(
                    result.Error,
                    Severity.Error,
                    static options =>
                    {
                        options.ShowCloseIcon = true;
                        options.VisibleStateDuration = 10_000;
                    });
            }
        }

        await GetDocumentsAsync();
    }

    private void OnShowDocumentAsync(DocumentSummary document)
    {
    }


    public void Dispose() => _cancellationTokenSource.Cancel();
}
