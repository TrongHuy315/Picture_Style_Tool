# Picture_Style_Tool

## Giới thiệu
Công cụ Picture_Style_Tool hỗ trợ chuyển đổi phong cách ảnh đầu vào cho người dùng
Hệ thống xử lý trích xuất đặc trưng và tái tạo phong cách ảnh bởi pretrained model `arbitrary-image-stylization-v1-256/2` của [Google](https://www.google.co.uk/)

Để hiểu rõ hơn về ứng dụng, hãy xem qua bảng phân loại phong cách nghệ thuật tranh vẽ dưới đây:

| Style                              | Mô tả             |
|------------------------------------|-------------------|
|Classical Painting Styles           |[Link](https://www.wikiart.org/en/paintings-by-style/classicism#!#filterName:all-works,viewType:masonry)                   |
|Modern and Abstract Styles          |[Link](https://www.art-is-fun.com/modern-abstract-art)                   |
|Cultural and Folk Styles            |[Link](https://library.fiveable.me/lists/folk-art-styles)                   |
|Natural Styles                      |[Link](https://www.tate.org.uk/art/art-terms/n/naturalism)                   |
|Technological and Futuristic Styles |[Link](https://blog.depositphotos.com/retro-futurism-art-design.html)                   |
|Vintage and Retro Styles            |[Link](https://picsart.com/blog/introduction-to-retro-art/)                   |
|Movie and Story Inspired Styles     |[Link](https://entergallery.com/blogs/news/art-inspired-by-the-movies?srsltid=AfmBOor6dc8K_McxXXdi50Gj22Y8aqhAQay5OBGHfylo2G0ZuAWCjxe5)                   |

## Hướng dẫn
- Đầu tiên, cần phải download những thư viện cần thiết trong file requirements.txt với các phiên bản tương thích nhau.
- Kiểm tra đường dẫn pretrained model: `https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2` trước khi sử dụng.
- Thực hiện kiểm tra chương trình chạy trên local trước bằng lệnh: `streamlit app.py`
- Nếu đã thực hiện thành công ở local thì có thể push lên repository trên github để deploy trên streamlit cloud hoặc render (nếu muốn thực hiện việc configuration).
