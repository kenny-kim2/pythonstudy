$().ready(function(){
    // getProductData(2);

});

function predictData() {
    var input_data = {};
    input_data['image_url'] = $('#input_url').val();
    wrapWindowByMask();
    $.ajax({
        url:'/test',
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
            closeMask();
        },
        error:function(request, status, error){
            alert('에러 발생');
            console.log(request);
            console.log(status);
            console.log(error);
            closeMask();
        }
    });
}

function wrapWindowByMask(){
    //화면의 높이와 너비를 구한다.
    var maskHeight = $(document).height();
    var maskWidth = $(window).width();
    $('#mask').show();

    //마스크의 높이와 너비를 화면 것으로 만들어 전체 화면을 채운다.
    $('#mask').css({'width':maskWidth,'height':maskHeight});

    var top = maskHeight/2 - $('#mask_img').height();
    var left = maskWidth/2 - $('#mask_img').width();

    $('#mask_img').css({'position':'absolute'});
    $('#mask_img').css({'top':top,'left':left});

    //애니메이션 효과
    $('#mask').fadeIn(1000);
    $('#mask').fadeTo("slow",0.8);
}

function closeMask(){
    $('#mask, .window').hide();
}
