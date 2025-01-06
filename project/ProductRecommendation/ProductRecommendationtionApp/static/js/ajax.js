function addProduct() {
  $("#heading").text("Add Product Details");
  $("#addProductModal").modal("show");
}


function editProduct(id) {  
  $.ajax({
    type: "GET",
    url: "/getProductInfo/",
    data: {
      id: id,
    },
    dataType: "json",
    success: function (data) {
      console.log("Product Details:", data);

      $("#id").val(data.id);
      $("#name").val(data.name);
      $("#manufacturer").val(data.manufacturer);
      $("#qty").val(data.quantity);
      $("#category").val(data.category);
      $("#price").val(data.price);
      $("#description").val(data.description);
      $("#recommendation").val(data.recommendation);
      // $("#images").val(data.image);

      $("#editProductModal").modal("show");
    },
    error: function (error) {
      console.log("Error:", error);
    },
  });
}

function orderNow(id) {
  $.ajax({
    type: "GET",
    url: "/getBook/",
    data: {
      id: id,
    },
    dataType: "json",
    success: function (data) {
      console.log("Book Details:", data);

      $("#id").val(data.id);
      $("#title").val(data.title);
      $("#name").val(data.name);
      $("#author").val(data.author);
      $("#category").val(data.category);
      $("#price").val(data.price);
      $("#description").val(data.description);
      $("#recommendation").val(data.recommendation);
      $("#images").val(data.image);

      $("#editBookModal").modal("show");
    },
    error: function (error) {
      console.log("Error:", error);
    },
  });
  $("#heading").text("Order Book");
  $("#orderBookModal").modal("show");
}

function orderNow(id) {
  $("#heading").text("Order Product");
  // $("#orderProduct").modal("show");
  $.ajax({
    type: "GET",
    url: "/getProductInfo/",
    data: {
      id: id,
    },
    dataType: "json",
    success: function (data) {
      console.log("Product Details:", data);

      $("#id").val(data.id);
      $("#price").val(data.price);
      $("#total").val(data.price);

      $("#orderProduct").modal("show");
    },
    error: function (error) {
      console.log("Error:", error);
    },
  });
}
