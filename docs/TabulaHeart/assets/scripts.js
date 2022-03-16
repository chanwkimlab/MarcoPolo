/*!
    * Start Bootstrap - SB Admin v6.0.1 (https://startbootstrap.com/templates/sb-admin)
    * Copyright 2013-2020 Start Bootstrap
    * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-sb-admin/blob/master/LICENSE)
    */
    (function($) {
    "use strict";

    // Add active state to sidbar nav links
    var path = window.location.href; // because the 'href' property of the DOM element is the absolute path
        $("#layoutSidenav_nav .sb-sidenav a.nav-link").each(function() {
            if (this.href === path) {
                $(this).addClass("active");
            }
        });

    // Toggle the side navigation
    $("#sidebarToggle").on("click", function(e) {
        e.preventDefault();
        $("body").toggleClass("sb-sidenav-toggled");
    });
})(jQuery);

function htmlDecode(input){
  var e = document.createElement('textarea');
  e.innerHTML = input;
  // handle case of empty input
  return e.childNodes.length === 0 ? "" : e.childNodes[0].nodeValue;
}

function format ( d ) {
    // `d` is the original data object for the row
    return '<table cellpadding="5" cellspacing="0" border="0" style="padding-left:50px;">'+
        '<tr>'+
            '<td>'+htmlDecode(d[d.length-1])+'</td>'+
        '</tr>'+
    '</table>';
}


// Call the dataTables jQuery plugin
$(document).ready(function() {
    var table =$('#dataTable').DataTable({
                                            "pageLength": 50,
                                            "order": [[ 0, "asc" ]],
                                            "columnDefs": [{
                                                            "targets": [ -1 ],
                                                            "visible": false,
                                                            "searchable": false
                                                           }]
                                        });
    
    
$('#dataTable tbody').on('click', 'tr', function () {
    var tr = $(this).closest('tr');
    var row = table.row( tr );

    if ( row.child.isShown() ) {
        // This row is already open - close it
        row.child.hide();
        tr.removeClass('shown');
    }
    else {
        // Open this row
        //row.child('<img>dsdsd').show();//format(row.data()) 
        row.child(format(row.data()) ).show();//format(r) 
        tr.addClass('shown');
    }
} );    
    
});
  
