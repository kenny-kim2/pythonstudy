$().ready(function(){
    // getProductData(2);

});

function predictData() {
    var input_data = {};
    input_data['image_url'] = $('#input_url').val();    
    $.ajax({
        url:'/predict',
        cache:false,
        data:input_data,
        method:'POST',
        success:function(data){
            console.log(data);
            returndata = JSON.parse(data)
            if(returndata['has_cate']){
                $('#category_name').text(returndata['result_cate_text']);
                $('#input_image').attr('src', returndata['input_url']);
                console.log(returndata['result_prod_code'])

                if(returndata.hasOwnProperty('result_prod_code')){
                    if(parseInt(returndata['result_prod_code']) > 0){
                        productCode = parseInt(returndata['result_prod_code'])

                        if(productCode > 500000) {
                            var code = ""+productCode;
                            var codelen = code.length;

                            for(var inx=code.length;inx < 6; inx++) {
                                code="0"+code;
                            }

                            subImagePath1 = code.substring(codelen-3, codelen);
                            subImagePath2 = code.substring(codelen-6, codelen-3);
                            imageUrl = "http://img.danawa.com/prod_img/500000/"+subImagePath1+"/"+subImagePath2+"/img/"+productCode+"_1_80.jpg";
                        } else {
                            imageUrl = "http://img.danawa.com/prod_img/small/group_"+Math.floor(productCode/500)+"/"+productCode+"_1.jpg";
                        }

                        $('#output_image').attr('src', imageUrl);
                        $('#output_code').text(productCode);
                    } else {
                        alert('기준상품 없음');
                    }
                } else {
                    alert('기준상품 없음');
                }
            } else {
                alert('맞는 카테고리가 없습니다.');
                $('#input_image').attr('src', returndata['input_url']);
            }

            alet('예측 완료!');
        },
        error:function(request, status, error){
            alert('에러 발생');
            console.log(request);
            console.log(status);
            console.log(error);
        }
    })
}
