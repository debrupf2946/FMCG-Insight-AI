import google.generativeai as genai

genai.configure(api_key="AIzaSyBwLyqi29z_tAU9_vXfTz9CkLzHgxpVX1A")

sample_file = genai.upload_file(path="qwen_images_1/30_1729295860.jpg",
                                display_name="Jetpack drawing")
print("File Uploaded")

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

response = model.generate_content([sample_file, "from the given  product image,perform OCR and try to find out BRAND name,EXIRY_DATE,MRP,SIZE,the product image may be blurry or hand might be there but intelligenty find it,out put in structured manner. Donot output anything if you donot detect any fmcg product or fruit or vegitable"])

print(">" + response.text)