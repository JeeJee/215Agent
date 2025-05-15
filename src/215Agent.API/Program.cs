using Microsoft.AspNetCore.Mvc;

/// <summary>
/// Initializes a new instance of the <see cref="HttpClient"/> class for sending HTTP 
/// requests and receiving HTTP responses from a resource identified by a URI.
/// </summary>
using HttpClient client = new();

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new Microsoft.OpenApi.Models.OpenApiInfo
    {
        Title = "215Agent API",
        Version = "v1"
    });
});

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "215Agent API V1");
    c.RoutePrefix = string.Empty; // Set Swagger UI at the app's root
});

// Endpoint 1: Text generation (RAG core)
app.MapPost("/api/rag", async ([FromBody] RAGRequest request) =>
{
    if (string.IsNullOrEmpty(request.Prompt))
        return Results.BadRequest("Prompt is required.");

    var ollamaRequest = new
    {
        prompt = request.Prompt,
        model = request.Model ?? "mistral",
        stream = false
    };
    var response = await client.PostAsJsonAsync("http://localhost:11434/api/generate", ollamaRequest);

    if (response.IsSuccessStatusCode)
    {
        var result = await response.Content.ReadAsStringAsync();
        return Results.Content(result, "application/json");
    }
    else
    {
        return Results.StatusCode((int)response.StatusCode);
    }
});
// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

// ✅ Endpoint 2: Text embedding
app.MapPost("/api/embed", async ([FromBody] EmbedRequest req) =>
{
    if (string.IsNullOrWhiteSpace(req.Text))
        return Results.BadRequest("Text is required.");

    var embedPayload = new
    {
        model = "nomic-embed-text",
        prompt = req.Text
    };

    var response = await client.PostAsJsonAsync("http://localhost:11434/api/embeddings", embedPayload);

    if (!response.IsSuccessStatusCode)
        return Results.Problem("Embedding request failed.");

    var json = await response.Content.ReadAsStringAsync();
    return Results.Content(json, "application/json");
});

app.Run();

public record RAGRequest(string Prompt, string? Model);
public record EmbedRequest(string Text);
