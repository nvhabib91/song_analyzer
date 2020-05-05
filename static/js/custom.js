$(function () {
    var getData = function (request, response) {
        $.getJSON(
            "_autocomp/" + request.term,
            function (data) {
                console.log(data)
                var array = $.map(data, function (m) {
                    return {
                        label: m.TitleArtist,
                        url: m.URL
                    };
                });
                response(array);
            });
    };

    var selectItem = function (event, ui) {
        console.log("Select item",ui);
        // event.preventDefault();
        $(this).text(ui.item.label);
        $("#songUrl").attr("value",ui.item.url)
        // $('form[name=searchSong]').attr('action', ui.item.url);
        //window.open(ui.item.url);
        // return false;
    }

    $("#autocomplete").autocomplete({
        source: getData,
        select: selectItem,
        minLength: 4
    });
});

