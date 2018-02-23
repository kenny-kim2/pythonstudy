$().ready(function(){
    getProductData(2);
});

function getProductData(page) {
    var pagedata = {};
    pagedata['page'] = page;
    $.ajax({
        url:'/product',
        cache:false,
        data:pagedata,
        method:'POST',
        success:function(data){
            console.log(data);
            returndata = JSON.parse(data)

            $('#prod_list').empty();
            var prodstring = '';
            var count = 0;
            for(var key in returndata){
                // <div class="row" >
                //     <div class="col-sm-2">
                //         <a data-toggle="modal" data-target="#myModal">
                //             <img src="http://i.011st.com/t/300/pd/17/7/8/6/8/7/0/zUvoE/1406786870_B.jpg" alt="Rounded Image" class="img-rounded img-responsive">
                //             <h4>링크 예상 수량</h4>
                //         </a>
                //     </div>

                if(count % 6 == 0){
                    prodstring += '<div class="row" >';
                }
                prodstring += ' <div class="col-sm-2">';
                prodstring += '     <a data-toggle="modal" data-target="#myModal">';
                prodstring += '         <img src="'+returndata[key]['img_url']+'" alt="Rounded Image" class="img-rounded img-responsive">';
                prodstring += '         <h4>'+returndata[key]['prod_name']+'('+returndata[key]['link_cnt']+')</h4>';
                prodstring += '     </a>';
                prodstring += ' </div>';
                if(count % 6 == 5){
                    prodstring += '</div>';
                }
                count++;
            }
            $('#prod_list').append(prodstring);

        },
        error:function(request, status, error){
            alert('실패');
            console.log(request);
            console.log(status);
            console.log(error);
        }
    })
}
