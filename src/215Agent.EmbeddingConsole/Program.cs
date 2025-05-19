// See https://aka.ms/new-console-template for more information
using System.Net.Http.Json;
using System.Text.Json;

Console.WriteLine("Hello, World!");

using HttpClient http = new();
string ollamaUrl = "http://localhost:11434/api/embeddings";
string qdrantUrl = "http://localhost:6333/collections/documents/points";

await CreateCollectionAsync();
Console.WriteLine("📂 Enter path to folder containing .txt or .md files:");
var folderPath = Console.ReadLine();

if (!Directory.Exists(folderPath)) {
	Console.WriteLine("❌ Folder not found.");
	return;
}

var files = Directory.GetFiles(folderPath, "*.*", SearchOption.TopDirectoryOnly);

foreach (var file in files)
{
	if (!file.EndsWith(".txt") && !file.EndsWith(".md")) continue;

	var content = await File.ReadAllTextAsync(file);
	var chunks = ChunkText(content, 1000);
	var filename = Path.GetFileName(file);

	for (int i = 0; i < chunks.Count; i++)
	{
		var chunk = chunks[i];
		var embedding = await GetEmbeddingAsync(chunk);

		if (embedding != null)
			await SendToQdrantAsync(embedding, chunk, filename, i);
	}
}

Console.WriteLine("✅ Done embedding and uploading all files.");


List<string> ChunkText(string text, int maxLength)
{
	var chunks = new List<string>();
	for (int i = 0; i < text.Length; i += maxLength)
	{
		int len = Math.Min(maxLength, text.Length - i);
		chunks.Add(text.Substring(i, len));
	}
	return chunks;
}

async Task<List<float>?> GetEmbeddingAsync(string text)
{
	var payload = new
	{
		model = "nomic-embed-text",
		prompt = text
	};

	try
	{
		var response = await http.PostAsJsonAsync(ollamaUrl, payload);
		var json = await response.Content.ReadFromJsonAsync<JsonElement>();
		return json.GetProperty("embedding").Deserialize<List<float>>();
	}
	catch (Exception ex)
	{
		Console.WriteLine($"⚠️ Failed to embed: {ex.Message}");
		return null;
	}
}

async Task SendToQdrantAsync(List<float> vector, string text, string source, int chunkIndex)
{
	var payload = new
	{
		points = new[]
		{
			new
			{
				id = Guid.NewGuid().ToString(),
				vector = vector,
				payload = new
				{
					text = text,
					source = source,
					chunk_index = chunkIndex
				}
			}
		}
	};

	try
	{
		var response = await http.PutAsJsonAsync(qdrantUrl, payload);
		if (!response.IsSuccessStatusCode)
			Console.WriteLine($"❌ Qdrant error: {response.StatusCode}");
		else
			Console.WriteLine($"✅ Sent to Qdrant: {source} (chunk {chunkIndex})");
	}
	catch (Exception ex)
	{
		Console.WriteLine($"❌ Failed to send to Qdrant: {ex.Message}");
	}
}

async Task CreateCollectionAsync()
{
	var payload = new
	{
		vectors = new
		{
			size = 768,
			distance = "Cosine"
		}
	};

	var response = await http.PutAsJsonAsync("http://localhost:6333/collections/documents", payload);

	var json = await response.Content.ReadAsStringAsync();
	Console.WriteLine($"Response: {json}");
}
